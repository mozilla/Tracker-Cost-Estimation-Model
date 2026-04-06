-- Temporal holdout: September 2024 per-request features
-- Same table as June query (crawl.requests), but third_parties uses June date
-- (the almanac table may not have a September entry)
--
-- Save to: data/raw/per_request_1pct_sep2024.csv

SELECT
  NET.HOST(req.url) AS tracker_domain,

  REGEXP_EXTRACT(req.url, r'https?://[^/]+(\/[^?#]*)') AS url_path,
  ARRAY_LENGTH(SPLIT(COALESCE(REGEXP_EXTRACT(req.url, r'https?://[^/]+(\/[^?#]*)'), '/'), '/')) - 1 AS path_depth,
  LOWER(REGEXP_EXTRACT(req.url, r'\.([a-zA-Z0-9]+)(?:\?|#|$)')) AS file_extension,
  REGEXP_CONTAINS(req.url, r'\?') AS has_query_params,
  LENGTH(req.url) AS url_length,
  ARRAY_LENGTH(SPLIT(COALESCE(REGEXP_EXTRACT(req.url, r'\?(.*)$'), ''), '&')) AS num_query_params,

  req.type AS resource_type,

  JSON_VALUE(req.payload, '$._initiator_type') AS initiator_type,
  JSON_VALUE(req.payload, '$._priority') AS chrome_priority,
  JSON_VALUE(req.payload, '$._method') AS http_method,
  JSON_VALUE(req.payload, '$._protocol') AS http_version,
  CAST(JSON_VALUE(req.payload, '$._is_secure') AS INT64) AS is_https,
  req.index AS waterfall_index,

  NET.HOST(req.page) AS page_domain,

  CAST(JSON_VALUE(req.payload, '$._bytesIn') AS INT64) AS transfer_bytes,
  CAST(JSON_VALUE(req.payload, '$._objectSizeUncompressed') AS INT64) AS uncompressed_bytes,
  CAST(JSON_VALUE(req.payload, '$._load_ms') AS INT64) AS load_ms,
  CAST(JSON_VALUE(req.payload, '$._ttfb_ms') AS INT64) AS ttfb_ms,
  CAST(JSON_VALUE(req.payload, '$._download_ms') AS INT64) AS download_ms,
  JSON_VALUE(req.payload, '$._contentType') AS content_type,
  JSON_VALUE(req.payload, '$._contentEncoding') AS content_encoding,

FROM `httparchive.crawl.requests` req
WHERE req.date = '2024-09-01'
  AND req.client = 'mobile'
  AND req.is_root_page = TRUE
  AND NET.HOST(req.url) IN (
    SELECT domain
    FROM `httparchive.almanac.third_parties`
    WHERE date = '2024-06-01'
      AND category IN ('ad', 'analytics', 'social', 'tag-manager', 'consent-provider')
  )
  AND MOD(ABS(FARM_FINGERPRINT(CONCAT(req.page, req.url))), 100) = 0
