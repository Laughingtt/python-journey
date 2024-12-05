

https://docs.starrocks.io/zh/docs/using_starrocks/caching/block_cache/

是否开启data_cache
执行以下语句查看是否已启用 Data Cache： starlet_use_star_cache

```sql
SELECT * FROM information_schema.be_configs 
WHERE NAME LIKE "%starlet_use_star_cache%";
```

禁用
```sql
UPDATE information_schema.be_configs SET VALUE = 0 
WHERE name = "starlet_use_star_cache";
```


Data Cache 的磁盘使用上限
starlet_star_cache_disk_size_percent

```sql
UPDATE information_schema.be_configs SET VALUE = 20 
WHERE name = "starlet_star_cache_disk_size_percent";
```