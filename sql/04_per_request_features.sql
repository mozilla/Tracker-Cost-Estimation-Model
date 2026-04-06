-- Per-request features for tracker domains
-- Goal: test whether URL path + resource type + initiator predict response size
--       better than a domain-level average (lookup table baseline)
--
-- This pulls individual requests (not aggregated) with:
--   Features (available before response / at block time):
--     - domain, URL path, resource type, initiator type, waterfall position
--   Labels (only available after response completes):
--     - transfer size, load time, TTFB
--
-- We sample to keep costs manageable:
--   - Top 200 highest-variance tracker domains (by p90/p50 transfer ratio)
--   - Plus top 50 most common tracker domains (for coverage)
--   - Up to 5000 requests per domain
--
-- Estimated cost: ~$3-8 (scans requests table filtered by domain list)

WITH tracker_domains AS (
  SELECT domain
  FROM `httparchive.almanac.third_parties`
  WHERE date = '2024-06-01'
    AND category IN ('ad', 'analytics', 'social', 'tag-manager', 'consent-provider')
),

-- First pass: identify high-variance and high-prevalence domains
domain_stats AS (
  SELECT
    NET.HOST(req.url) AS tracker_domain,
    COUNT(*) AS total_requests,
    COUNT(DISTINCT req.page) AS pages_seen_on,
    APPROX_QUANTILES(CAST(JSON_VALUE(req.payload, '$._bytesIn') AS INT64), 100)[OFFSET(50)] AS p50_bytes,
    APPROX_QUANTILES(CAST(JSON_VALUE(req.payload, '$._bytesIn') AS INT64), 100)[OFFSET(90)] AS p90_bytes,
  FROM `httparchive.crawl.requests` req
  WHERE req.date = '2024-06-01'
    AND req.client = 'mobile'
    AND req.is_root_page = TRUE
    AND NET.HOST(req.url) IN (SELECT domain FROM tracker_domains)
  GROUP BY tracker_domain
  HAVING COUNT(*) >= 100
),

-- Select domains worth studying: high variance OR high prevalence
selected_domains AS (
  SELECT tracker_domain FROM (
    -- High variance domains (p90/p50 ratio > 5)
    SELECT tracker_domain
    FROM domain_stats
    WHERE p50_bytes > 0 AND (p90_bytes / p50_bytes) > 5
    ORDER BY (p90_bytes / p50_bytes) DESC
    LIMIT 200
  )

  UNION DISTINCT

  SELECT tracker_domain FROM (
    -- Most common domains (top 50 by page count)
    SELECT tracker_domain
    FROM domain_stats
    ORDER BY pages_seen_on DESC
    LIMIT 50
  )
)

-- Pull per-request data for selected domains
SELECT
  NET.HOST(req.url) AS tracker_domain,

  -- URL features (available at block time)
  -- Extract path without query params for pattern learning
  REGEXP_EXTRACT(req.url, r'https?://[^/]+(\/[^?#]*)') AS url_path,
  -- Path depth (number of / segments)
  ARRAY_LENGTH(SPLIT(REGEXP_EXTRACT(req.url, r'https?://[^/]+(\/[^?#]*)'), '/')) - 1 AS path_depth,
  -- File extension
  REGEXP_EXTRACT(req.url, r'\.([a-zA-Z0-9]+)(?:\?|#|$)') AS file_extension,
  -- Has query parameters
  REGEXP_CONTAINS(req.url, r'\?') AS has_query_params,
  -- URL length
  LENGTH(req.url) AS url_length,

  -- Resource type (available at block time)
  req.type AS resource_type,

  -- Request metadata (available at block time)
  JSON_VALUE(req.payload, '$._initiator_type') AS initiator_type,
  JSON_VALUE(req.payload, '$._priority') AS chrome_priority,
  JSON_VALUE(req.payload, '$._method') AS http_method,
  JSON_VALUE(req.payload, '$._protocol') AS http_version,
  req.index AS waterfall_index,

  -- Page context (available at block time)
  NET.HOST(req.page) AS page_domain,

  -- LABELS (NOT available at block time -- this is what we predict)
  CAST(JSON_VALUE(req.payload, '$._bytesIn') AS INT64) AS transfer_bytes,
  CAST(JSON_VALUE(req.payload, '$._objectSizeUncompressed') AS INT64) AS uncompressed_bytes,
  CAST(JSON_VALUE(req.payload, '$._load_ms') AS INT64) AS load_ms,
  CAST(JSON_VALUE(req.payload, '$._ttfb_ms') AS INT64) AS ttfb_ms,
  CAST(JSON_VALUE(req.payload, '$._download_ms') AS INT64) AS download_ms,
  JSON_VALUE(req.payload, '$._contentType') AS content_type,
  JSON_VALUE(req.payload, '$._contentEncoding') AS content_encoding,

FROM `httparchive.crawl.requests` req
WHERE req.date = '2024-06-01'
  AND req.client = 'mobile'
  AND req.is_root_page = TRUE
  AND NET.HOST(req.url) IN (SELECT tracker_domain FROM selected_domains)
  -- Sample up to ~5000 requests per domain via hash
  AND MOD(ABS(FARM_FINGERPRINT(req.url)), 100) < 5
ORDER BY tracker_domain, req.page
