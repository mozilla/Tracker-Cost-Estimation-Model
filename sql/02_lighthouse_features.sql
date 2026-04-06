SELECT                                                                                                                                                                      
  'bootup' AS source,                                                                                                                                                       
  NET.HOST(JSON_VALUE(item, '$.url')) AS tracker_domain,                                                                                                                    
  COUNT(*) AS occurrences,                                                                                                                                                  
  COUNT(DISTINCT page) AS pages_with_data,                                                                                                                                
  AVG(CAST(JSON_VALUE(item, '$.scripting') AS FLOAT64)) AS mean_scripting_ms,                                                                                               
  AVG(CAST(JSON_VALUE(item, '$.scriptParseCompile') AS FLOAT64)) AS mean_parse_compile_ms,                                                                                  
  AVG(CAST(JSON_VALUE(item, '$.total') AS FLOAT64)) AS mean_total_cpu_ms,                                                                                                   
  NULL AS mean_main_thread_ms,                                                                                                                                              
  NULL AS mean_transfer_size,                                                                                                                                               
  NULL AS mean_wasted_ms,                                                                                                                                                 
  NULL AS mean_total_bytes,                                                                                                                                                 
FROM `httparchive.crawl.pages`,                                                                                                                                           
  UNNEST(JSON_QUERY_ARRAY(lighthouse,                                                                                                                                       
    '$.audits.bootup-time.details.items')) AS item                                                                                                                          
WHERE date = '2024-06-01'                                                                                                                                                   
  AND client = 'mobile'                                                                                                                                                     
  AND is_root_page = TRUE                                                                                                                                                   
GROUP BY tracker_domain                                                                                                                                                     
                                                                                                                                                                          
UNION ALL                                                                                                                                                                   
                                                                                                                                                                          
SELECT                                                                                                                                                                    
  'third_party' AS source,
  NULL AS tracker_domain,
  COUNT(*) AS occurrences,
  COUNT(DISTINCT page) AS pages_with_data,                                                                                                                                  
  NULL AS mean_scripting_ms,
  NULL AS mean_parse_compile_ms,                                                                                                                                            
  NULL AS mean_total_cpu_ms,                                                                                                                                              
  AVG(CAST(JSON_VALUE(item, '$.mainThreadTime') AS FLOAT64)) AS mean_main_thread_ms,                                                                                        
  AVG(CAST(JSON_VALUE(item, '$.transferSize') AS FLOAT64)) AS mean_transfer_size,
  NULL AS mean_wasted_ms,                                                                                                                                                   
  NULL AS mean_total_bytes,                                                                                                                                               
FROM `httparchive.crawl.pages`,                                                                                                                                             
  UNNEST(JSON_QUERY_ARRAY(lighthouse,                                                                                                                                     
    '$.audits.third-parties-insight.details.items')) AS item                                                                                                              
WHERE date = '2024-06-01'                                                                                                                                                   
  AND client = 'mobile'
  AND is_root_page = TRUE                                                                                                                                                   
GROUP BY JSON_VALUE(item, '$.entity')                                                                                                                                       
                                                                                                                                                                          
UNION ALL                                                                                                                                                                   
                                                                                                                                                                          
SELECT                                                                                                                                                                    
  'render_blocking' AS source,
  NET.HOST(JSON_VALUE(item, '$.url')) AS tracker_domain,
  COUNT(*) AS occurrences,                                                                                                                                                  
  COUNT(DISTINCT page) AS pages_with_data,
  NULL AS mean_scripting_ms,                                                                                                                                        
  NULL AS mean_parse_compile_ms,                                                                                                                                          
  NULL AS mean_total_cpu_ms,                                                                                                                                                
  NULL AS mean_main_thread_ms,
  NULL AS mean_transfer_size,                                                                                                                                               
  AVG(CAST(JSON_VALUE(item, '$.wastedMs') AS FLOAT64)) AS mean_wasted_ms,                                                                                                 
  AVG(CAST(JSON_VALUE(item, '$.totalBytes') AS FLOAT64)) AS mean_total_bytes,                                                                                               
FROM `httparchive.crawl.pages`,
  UNNEST(JSON_QUERY_ARRAY(lighthouse,                                                                                                                                       
    '$.audits.render-blocking-insight.details.items')) AS item                                                                                                            
WHERE date = '2024-06-01'                                                                                                                                                   
  AND client = 'mobile'                                                                                                                                                   
  AND is_root_page = TRUE                                                                                                                                                 
GROUP BY tracker_domain  