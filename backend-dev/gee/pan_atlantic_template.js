// gee/pan_atlantic_template.js
// Template GEE script for pan-Atlantic inference regions.
// Swap REGION_NAME and REGION_POLYGON for each target, then Run.
// Output: <REGION_NAME>_s2_pixels.csv in Drive folder 'EelgrassEEWS'.

// ─── Region configuration ─────────────────────────────────────────────────
var REGION_NAME = 'antigonish';              // <-- change for each region
var DATE_START  = '2023-01-01';              // last 90+ days of valid windows
var DATE_END    = '2024-12-31';

// Coordinates must match a polygon in regions.json
var REGION_POLYGON = ee.Geometry.Polygon([[
  [-61.94, 45.61], [-61.88, 45.61], [-61.88, 45.65], [-61.94, 45.65], [-61.94, 45.61]
]]);
var SITE_LABEL = 'Antigonish_Main';

// ─── S2 collection ────────────────────────────────────────────────────────
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(REGION_POLYGON)
  .filterDate(DATE_START, DATE_END)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20));

print('Image count for ' + REGION_NAME + ':', s2.size());

// ─── Same index function as forillon_extraction.js ────────────────────────
function addIndices(image) {
  var s = 10000;
  var B2  = image.select('B2').divide(s);
  var B3  = image.select('B3').divide(s);
  var B4  = image.select('B4').divide(s);
  var B5  = image.select('B5').divide(s);
  var B6  = image.select('B6').divide(s);
  var B8  = image.select('B8').divide(s);
  var B8A = image.select('B8A').divide(s);
  var B11 = image.select('B11').divide(s);
  var B12 = image.select('B12').divide(s);
  var eps = ee.Image(1e-6);

  return image
    .select(['B2','B3','B4','B5','B6','B8','B8A','B11','B12'])
    .divide(s)
    .rename(['B2','B3','B4','B5','B6','B8','B8A','B11','B12'])
    .addBands([
      B8.subtract(B4).divide(B8.add(B4).add(eps)).rename('NDAVI'),
      B8.subtract(B4).multiply(1.5).divide(B8.add(B4).add(0.5)).rename('WAVI'),
      B3.divide(B2.add(eps)).rename('GB_ratio'),
      B4.divide(B3.add(eps)).rename('RG_ratio'),
      B3.subtract(B2).rename('B3B2_diff'),
      B3.subtract(B8).divide(B3.add(B8).add(eps)).rename('NDWI'),
      B4.divide(B3.add(eps)).rename('turbidity'),
      B6.subtract(B5).divide(35).rename('red_edge_slope'),
      B8.subtract(B4).divide(B3.add(B2).add(eps)).rename('SABI'),
      B2.log().divide(B3.log().add(eps)).rename('depth_invariant'),
    ])
    .set('system:time_start', image.get('system:time_start'))
    .set('CLOUDY_PIXEL_PERCENTAGE', image.get('CLOUDY_PIXEL_PERCENTAGE'));
}

// Optional shallow-water mask: keep pixels where NDWI > 0 and NIR is low
// (avoid land + open deep water). Apply per image after indexing.
function maskWater(image) {
  var ndwi = image.select('NDWI');
  var nir  = image.select('B8');
  var mask = ndwi.gt(0).and(nir.lt(0.1));
  return image.updateMask(mask);
}

var s2_indexed = s2.map(addIndices).map(maskWater);

// ─── Sample pixels at 10 m ────────────────────────────────────────────────
var allFeatures = ee.FeatureCollection(
  s2_indexed.map(function(image) {
    var date  = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd');
    var cloud = image.get('CLOUDY_PIXEL_PERCENTAGE');
    return image.addBands(ee.Image.pixelLonLat())
      .sample({region: REGION_POLYGON, scale: 10, projection: 'EPSG:4326',
               geometries: false, dropNulls: true, seed: 42})
      .map(function(f) {
        return f.set('SITE', SITE_LABEL).set('DATE', date).set('CLOUD_PCT', cloud);
      });
  }).flatten()
);

print('Total pixel-image records:', allFeatures.size());

Export.table.toDrive({
  collection: allFeatures,
  description: REGION_NAME + '_s2_pixels',
  folder: 'EelgrassEEWS',
  fileNamePrefix: REGION_NAME + '_s2_pixels',
  fileFormat: 'CSV',
  selectors: ['SITE', 'DATE', 'CLOUD_PCT', 'longitude', 'latitude',
              'B2','B3','B4','B5','B6','B8','B8A','B11','B12',
              'NDAVI','WAVI','GB_ratio','RG_ratio','B3B2_diff',
              'NDWI','turbidity','red_edge_slope','SABI','depth_invariant']
});
