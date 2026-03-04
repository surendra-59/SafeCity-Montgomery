import pandas as pd
import geopandas as gpd

print("Loading Comma Separated Values files...")

# 1. Load the original datasets
requests_dataframe = pd.read_csv('311_requests_50k.csv')
violations_dataframe = pd.read_csv('code_violations_50k.csv')

# 2. Process the 311 Service Requests
# Filter specifically for water, drainage, and general nuisance issues
target_311_types = ['Drains', 'Nuisance', 'Clean Inlets', 'General Sanitation Inquiry']
filtered_311 = requests_dataframe[requests_dataframe['Request_Type'].isin(target_311_types)].copy()

# Convert the timestamp from milliseconds into a standard date format
filtered_311['date'] = pd.to_datetime(filtered_311['Create_Date'], unit='ms')

# Extract only the columns we need and standardize their names
clean_311 = pd.DataFrame({
    'issue_type': filtered_311['Request_Type'],
    'latitude': filtered_311['Latitude'],
    'longitude': filtered_311['Longitude'],
    'date': filtered_311['date'],
    'data_source': '311_Service_Request'
})

# 3. Process the Code Violations
# Filter specifically for overgrown properties, nuisances, and open/vacant buildings
target_code_types = ['NUISANCE', 'OPEN VACANT', 'REPAIR']
filtered_code = violations_dataframe[violations_dataframe['CaseType'].isin(target_code_types)].copy()

# Convert the text dates into standard datetime objects
filtered_code['date'] = pd.to_datetime(filtered_code['CaseDate'])

# Extract needed columns (Code violations use 'y' for latitude and 'x' for longitude)
clean_code = pd.DataFrame({
    'issue_type': filtered_code['CaseType'],
    'latitude': filtered_code['y'],
    'longitude': filtered_code['x'],
    'date': filtered_code['date'],
    'data_source': 'Code_Violation'
})

# 4. Merge both standardized dataframes into one unified dataset
combined_dataframe = pd.concat([clean_311, clean_code], ignore_index=True)

# 5. Drop any rows where the geographic coordinates are missing to prevent model errors
final_dataframe = combined_dataframe.dropna(subset=['latitude', 'longitude'])

print(f"\nData successfully merged!")
print(f"You now have {len(final_dataframe)} historical hazard points ready for processing.")
print("\nHere is a preview of your clean, unified data:")
print(final_dataframe.head())

print("\nSaving processed data to your project folder...")

print("\nLoading Weather Sirens to create Risk Zones...")
sirens_dataframe = pd.read_csv('Weather_Sirens.csv')

# 6. Convert our unified data into a Geographic DataFrame (spatial data)
incidents_gdf = gpd.GeoDataFrame(
    final_dataframe, 
    geometry=gpd.points_from_xy(final_dataframe.longitude, final_dataframe.latitude),
    crs="EPSG:4326" # Standard coordinate reference system for GPS
)

# 7. Convert the Sirens into a Geographic DataFrame
sirens_gdf = gpd.GeoDataFrame(
    sirens_dataframe,
    geometry=gpd.points_from_xy(sirens_dataframe.X, sirens_dataframe.Y),
    crs="EPSG:4326"
)

print("Assigning all 57,000+ incidents to their nearest siren (this may take a few seconds)...")
# 8. Spatial Join: Link each incident to the closest weather siren
clustered_data = gpd.sjoin_nearest(
    incidents_gdf, 
    sirens_gdf[['USER_Siren_Number', 'USER_Street_Address', 'geometry']], 
    how="left", 
    distance_col="distance_to_siren"
)

# 9. Group the data by siren to find the most historically dangerous zones
risk_zones = clustered_data.groupby(['USER_Siren_Number', 'USER_Street_Address']).size().reset_index(name='historical_incident_count')

# Sort to find the highest risk areas at the top
risk_zones = risk_zones.sort_values(by='historical_incident_count', ascending=False)

print("\n--- Top 5 Highest Risk Environmental Zones in Montgomery ---")
print(risk_zones.head(5))

# 10. Save the raw, merged incident data
final_dataframe.to_csv('cleaned_environmental_hazards.csv', index=False)

# 11. Save the newly calculated Risk Zones
risk_zones.to_csv('siren_risk_zones.csv', index=False)

print("Success! 'cleaned_environmental_hazards.csv' and 'siren_risk_zones.csv' have been saved.")