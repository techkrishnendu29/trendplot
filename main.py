from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse
import ee
import datetime
import os
import json
from io import BytesIO
import matplotlib.pyplot as plt

# ---------- Step 1: Authenticate Earth Engine ----------
serviceaccount = "gee-backend@ee-bairagisayan464.iam.gserviceaccount.com"
key_json = os.getenv("EE_KEY_JSON")

if key_json is None:
    raise Exception("Please set EE_KEY_JSON in Render Config Vars.")

key_dict = json.loads(key_json)

# Temporary file for EE credentials
with open("temp_key.json", "w") as f:
    json.dump(key_dict, f)

credentials = ee.ServiceAccountCredentials(serviceaccount, "temp_key.json")
ee.Initialize(credentials, project="ee-bairagisayan464")

# ---------- Step 2: FastAPI App ----------
app = FastAPI(title="NDVI Trend API", description="Google Earth Engine NDVI Time Series API", version="1.0")

# ---------- Step 3: Helper functions ----------
def s2Mask(img):
    qa = img.select("QA60")
    mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return img.updateMask(mask)

def growth(img):
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return img.addBands(ndvi).copyProperties(img, ["system:time_start"])

def extract_ndvi(image, geometry):
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=10,
        maxPixels=1e6,
        bestEffort=True
    )
    return ee.Feature(None, {
        'NDVI': stats.get('NDVI'),
        'date': image.date().format('YYYY-MM-dd')
    })

# ---------- Step 4: API Endpoint ----------
@app.get("/ndvi_trend")
def get_ndvi_trend(
    state: str = Query(..., description="State Name"),
    district: str = Query(..., description="District Name"),
    location: str = Query(..., description="Location Name"),
):
    try:
        # Dates
        end = datetime.date.today()
        start = end - datetime.timedelta(days=30)

        # Location filters
        country = ee.FeatureCollection("projects/ee-bairagisayan464/assets/GADM-IND").filter(
            ee.Filter.eq("COUNTRY", "India")
        )
        state_fc = country.filter(ee.Filter.eq("NAME_1", state))
        district_fc = state_fc.filter(ee.Filter.eq("NAME_2", district))
        location_fc = district_fc.filter(ee.Filter.eq("NAME_3", location))

        if location_fc.size().getInfo() == 0:
            raise HTTPException(status_code=404, detail="Location not found in GEE assets")

        # Image collection
        imgc = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(str(start), str(end))
            .filterBounds(location_fc)
            .map(s2Mask)
            .map(growth)
        )

        if imgc.size().getInfo() == 0:
            raise HTTPException(status_code=404, detail="No Sentinel-2 images found for this location and date range")

        # Time series extraction
        time_series = imgc.map(lambda img: extract_ndvi(img, district_fc.geometry())).getInfo()

        dates = []
        ndvi_values = []
        for feature in time_series['features']:
            props = feature['properties']
            ndvi = props.get('NDVI')
            if ndvi is not None:
                dates.append(props['date'])
                ndvi_values.append(ndvi)

        if not dates or not ndvi_values:
            raise HTTPException(status_code=404, detail="No NDVI values found for this location")

        # Plot NDVI Time Series
        plt.figure(figsize=(10, 5))
        plt.plot(dates, ndvi_values, marker='o', linestyle='-', color='blue')
        plt.title(f'NDVI Time Series (Last 30 Days)\n{location}, {district}, {state}')
        plt.xlabel('Date')
        plt.ylabel('NDVI')
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.tight_layout()

        # Save plot to memory buffer
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")

    except ee.EEException as e:
        raise HTTPException(status_code=500, detail=f"Earth Engine error: {e}")