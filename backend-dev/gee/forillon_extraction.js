// gee/forillon_extraction.js
// Paste into https://code.earthengine.google.com and Run.
// Output: forillon_s2_pixels.csv in Google Drive folder 'EelgrassEEWS'.

var ouest = ee.Geometry.Polygon([[
  [-64.458, 48.792], [-64.448, 48.792],
  [-64.448, 48.808], [-64.458, 48.808], [-64.458, 48.792]
]]);
var marais = ee.Geometry.Polygon([[
  [-64.438, 48.798], [-64.428, 48.798],
  [-64.428, 48.814], [-64.438, 48.814], [-64.438, 48.798]
]]);
var sud = ee.Geometry.Polygon([[
  [-64.443, 48.778], [-64.433, 48.778],
  [-64.433, 48.794], [-64.443, 48.794], [-64.443, 48.778]
]]);
var sites_dict = ee.Dictionary({'Ouest': ouest, 'Marais': marais, 'Sud': sud});
var roi = ouest.union(marais).union(sud);

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(roi)
  .filterDate('2015-06-01', '2018-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20));

print('Image count:', s2.size());

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

  var NDAVI = B8.subtract(B4).divide(B8.add(B4).add(eps)).rename('NDAVI');
  var WAVI  = B8.subtract(B4).multiply(1.5)
                .divide(B8.add(B4).add(0.5)).rename('WAVI');
  var GB    = B3.divide(B2.add(eps)).rename('GB_ratio');
  var RG    = B4.divide(B3.add(eps)).rename('RG_ratio');
  var B3B2  = B3.subtract(B2).rename('B3B2_diff');
  var NDWI  = B3.subtract(B8).divide(B3.add(B8).add(eps)).rename('NDWI');
  var TURB  = B4.divide(B3.add(eps)).rename('turbidity');
  var RES   = B6.subtract(B5).divide(35).rename('red_edge_slope');
  var SABI  = B8.subtract(B4).divide(B3.add(B2).add(eps)).rename('SABI');
  var DEPTH = B2.log().divide(B3.log().add(eps)).rename('depth_invariant');

  return image
    .select(['B2','B3','B4','B5','B6','B8','B8A','B11','B12'])
    .divide(s)
    .rename(['B2','B3','B4','B5','B6','B8','B8A','B11','B12'])
    .addBands([NDAVI, WAVI, GB, RG, B3B2, NDWI, TURB, RES, SABI, DEPTH])
    .set('system:time_start', image.get('system:time_start'))
    .set('CLOUDY_PIXEL_PERCENTAGE', image.get('CLOUDY_PIXEL_PERCENTAGE'));
}

var s2_indexed = s2.map(addIndices);

var siteNames = ['Ouest', 'Marais', 'Sud'];

var allFeatures = ee.FeatureCollection(
  s2_indexed.map(function(image) {
    var date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd');
    var cloud = image.get('CLOUDY_PIXEL_PERCENTAGE');
    var sitesFc = ee.List(siteNames).map(function(siteName) {
      var geom = ee.Geometry(sites_dict.get(siteName));
      return image.addBands(ee.Image.pixelLonLat())
        .sample({region: geom, scale: 10, projection: 'EPSG:4326',
                 geometries: false, dropNulls: true, seed: 42})
        .map(function(f) {
          return f.set('SITE', siteName).set('DATE', date).set('CLOUD_PCT', cloud);
        });
    });
    return ee.FeatureCollection(sitesFc).flatten();
  }).flatten()
);

print('Total pixel-image records:', allFeatures.size());

Export.table.toDrive({
  collection: allFeatures,
  description: 'forillon_s2_pixels',
  folder: 'EelgrassEEWS',
  fileNamePrefix: 'forillon_s2_pixels',
  fileFormat: 'CSV',
  selectors: ['SITE', 'DATE', 'CLOUD_PCT', 'longitude', 'latitude',
              'B2','B3','B4','B5','B6','B8','B8A','B11','B12',
              'NDAVI','WAVI','GB_ratio','RG_ratio','B3B2_diff',
              'NDWI','turbidity','red_edge_slope','SABI','depth_invariant']
});
