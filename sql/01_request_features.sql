-- Stage 1: Per-request features for all third-party tracker domains
-- Run against: httparchive.crawl.requests (2024-06-01, mobile)
-- Estimated cost: ~$2-5 (scans type + url columns, filtered by date/client)
-- Export result to: data/raw/request_features.parquet

SELECT
  NET.HOST(req.url) AS tracker_domain,
  req.page,
  req.type AS resource_type,
  CAST(JSON_VALUE(req.payload, '$._bytesIn') AS INT64) AS transfer_bytes,
  CAST(JSON_VALUE(req.payload, '$._objectSize') AS INT64) AS compressed_bytes,
  CAST(JSON_VALUE(req.payload, '$._load_ms') AS INT64) AS load_ms,
  CAST(JSON_VALUE(req.payload, '$._ttfb_ms') AS INT64) AS ttfb_ms,
  req.index AS waterfall_index,
  JSON_VALUE(req.payload, '$._priority') AS chrome_priority,
  JSON_VALUE(req.payload, '$._initiator_type') AS initiator_type,
  JSON_VALUE(req.payload, '$._contentType') AS content_type,
FROM `httparchive.crawl.requests` req
WHERE req.date = '2024-06-01'
  AND req.client = 'mobile'
  AND req.is_root_page = TRUE
  AND NET.HOST(req.url) IN (
    SELECT domain
    FROM `httparchive.almanac.third_parties`
    WHERE date = '2024-06-01'
      AND `category` IN ('ad', 'analytics', 'social', 'tag-manager', 'consent-provider')
  )
