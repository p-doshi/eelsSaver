def load_demo_hotspots():
    print("loading hotspots...", flush=True)

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-64.178, 44.614], [-64.145, 44.614], [-64.145, 44.632], [-64.178, 44.632], [-64.178, 44.614]
                    ]]
                },
                "properties": {
                    "id": "ns-001",
                    "name": "St. Margarets Bay",
                    "region": "Nova Scotia",
                    "country": "Canada",
                    "risk_level": "high",
                    "eelgrass_pct_estimate": 42.0,
                    "depletion_risk_90d": 0.74,
                    "confidence": 0.82,
                    "summary": "Spectral stress signature suggests reduced vegetation vigor and elevated near-term decline probability.",
                    "drivers": ["high turbidity anomaly", "seasonal vigor drop", "possible nutrient stress"],
                    "last_updated": "2026-05-23",
                    "reports": [
                        {"title": "Regional coastal vegetation pressure note", "url": "https://example.org/report/stmargarets"},
                        {"title": "Nearshore water quality summary", "url": "https://example.org/report/waterquality"}
                    ],
                    "imagery": {
                        "preview_url": "/static/overlays/ns-001-truecolor.webp",
                        "stress_overlay_url": "/static/overlays/ns-001-stress.png",
                        "risk_overlay_url": "/static/overlays/ns-001-risk.png",
                        "bounds": [[44.614, -64.178], [44.632, -64.145]],
                        "credit": "Sentinel-2 derived visualization"
                    }
                }
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-64.469, 48.913], [-64.412, 48.913], [-64.412, 48.944], [-64.469, 48.944], [-64.469, 48.913]
                    ]]
                },
                "properties": {
                    "id": "qc-001",
                    "name": "Forillon Anchor Site",
                    "region": "Québec",
                    "country": "Canada",
                    "risk_level": "moderate",
                    "eelgrass_pct_estimate": 58.0,
                    "depletion_risk_90d": 0.49,
                    "confidence": 0.91,
                    "summary": "Reference site used to align field observations with Sentinel-2-derived signals.",
                    "drivers": ["training anchor", "seasonal reference behavior", "higher label quality"],
                    "last_updated": "2026-05-23",
                    "reports": [
                        {"title": "Anchor site calibration summary", "url": "https://example.org/report/forillon-calibration"}
                    ],
                    "imagery": {
                        "preview_url": "/static/overlays/qc-001-truecolor.webp",
                        "stress_overlay_url": "/static/overlays/qc-001-stress.png",
                        "risk_overlay_url": "/static/overlays/qc-001-risk.png",
                        "bounds": [[48.913, -64.469], [48.944, -64.412]],
                        "credit": "Sentinel-2 derived visualization"
                    }
                }
            }
        ]
    }