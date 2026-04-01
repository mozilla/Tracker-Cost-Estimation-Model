-- Large-scale per-request features for overnight CNN training
-- 10% sample = ~35M rows
-- Export to GCS: gs://YOUR_BUCKET/per_request_10pct/*.csv
--
-- Run: bq query --use_legacy_sql=false --destination_table=YOUR_PROJECT.YOUR_DATASET.per_request_10pct < sql/06_large_scale.sql
-- Then: bq extract --destination_format=CSV YOUR_PROJECT.YOUR_DATASET.per_request_10pct gs://YOUR_BUCKET/per_request_10pct/*.csv
--
-- Estimated cost: ~$5-10

SELECT
  NET.HOST(req.url) AS tracker_domain,

  -- URL features (available at block time)
  REGEXP_EXTRACT(req.url, r'https?://[^/]+(\/[^?#]*)') AS url_path,
  ARRAY_LENGTH(SPLIT(COALESCE(REGEXP_EXTRACT(req.url, r'https?://[^/]+(\/[^?#]*)'), '/'), '/')) - 1 AS path_depth,
  LOWER(REGEXP_EXTRACT(req.url, r'\.([a-zA-Z0-9]+)(?:\?|#|$)')) AS file_extension,
  REGEXP_CONTAINS(req.url, r'\?') AS has_query_params,
  LENGTH(req.url) AS url_length,
  ARRAY_LENGTH(SPLIT(COALESCE(REGEXP_EXTRACT(req.url, r'\?(.*)$'), ''), '&')) AS num_query_params,

  -- Resource type (available at block time)
  req.type AS resource_type,

  -- Request metadata (available at block time)
  JSON_VALUE(req.payload, '$._initiator_type') AS initiator_type,
  JSON_VALUE(req.payload, '$._method') AS http_method,
  req.index AS waterfall_index,

  -- Page context (available at block time)
  NET.HOST(req.page) AS page_domain,

  -- LABELS (NOT available at block time)
  CAST(JSON_VALUE(req.payload, '$._bytesIn') AS INT64) AS transfer_bytes,
  CAST(JSON_VALUE(req.payload, '$._load_ms') AS INT64) AS load_ms,

FROM `httparchive.crawl.requests` req
WHERE req.date = '2024-06-01'
  AND req.client = 'mobile'
  AND req.is_root_page = TRUE
  AND NET.HOST(req.url) IN (
    SELECT domain
    FROM `httparchive.almanac.third_parties`
    WHERE date = '2024-06-01'
      AND category IN ('ad', 'analytics', 'social', 'tag-manager', 'consent-provider')
  )
  -- 10% sample (~35M rows)
  AND MOD(ABS(FARM_FINGERPRINT(CONCAT(req.page, req.url))), 10) = 0
