#!/usr/bin/env bash
# Probe free data sources from a GitHub runner (the dev workspace cannot reach them).
set +e
hr(){ echo; echo "=================== $1"; }
hr "CPC GIS us_tempprcpfcst listing"
curl -s -m 60 https://ftp.cpc.ncep.noaa.gov/GIS/us_tempprcpfcst/ -o cpc_gis.html; wc -c cpc_gis.html
grep -oE 'href="[^"]+"' cpc_gis.html | sed 's/href=//;s/"//g' | sed -E 's/_[0-9]{8}/_DATE/' | sort | uniq -c | sort -rn | head -30
for p in 610temp 814temp 610prcp 814prcp; do
  echo "$p first/last:"; grep -oE "${p}_[0-9]{8}\.zip" cpc_gis.html | sort -u | sed -n '1p;$p'; grep -oE "${p}_[0-9]{8}\.zip" cpc_gis.html | sort -u | wc -l; done
grep -oE 'href="[^"]+/"' cpc_gis.html | head -20
hr "CPC ftp top"
curl -s -m 60 https://ftp.cpc.ncep.noaa.gov/ | grep -oE 'href="[^"]+"' | head -60
hr "CPC GIS top"
curl -s -m 60 https://ftp.cpc.ncep.noaa.gov/GIS/ | grep -oE 'href="[^"]+"' | head -60
hr "CPC short_range archive dir"
for u in https://www.cpc.ncep.noaa.gov/products/archives/short_range/ https://www.cpc.ncep.noaa.gov/products/archives/short_range/2010/ https://www.cpc.ncep.noaa.gov/products/archives/short_range/2010/01/ ; do echo "-- $u"; curl -s -m 60 "$u" | grep -oE 'href="[^"]+"' | head -40; done
hr "CPC short_range php form target"
curl -s -m 60 https://www.cpc.ncep.noaa.gov/products/archives/short_range/srarc.ind.php | grep -iE 'form|action|input|\.php' | head -20
hr "CPC 610 page links to data/text"
curl -s -m 60 https://www.cpc.ncep.noaa.gov/products/predictions/610day/ | grep -oE 'href="[^"]+"' | grep -viE 'facebook|twitter' | head -80
hr "CPC verification / data dirs"
for u in https://ftp.cpc.ncep.noaa.gov/products/ https://www.cpc.ncep.noaa.gov/products/predictions/610day/archive/ https://www.cpc.ncep.noaa.gov/products/archives/ ; do echo "-- $u"; curl -s -m 60 "$u" | grep -oE 'href="[^"]+"' | head -40; done
hr "GEFS retrospective (reforecast v12)"
aws s3 ls --no-sign-request s3://noaa-gefs-retrospective/ | head
aws s3 ls --no-sign-request s3://noaa-gefs-retrospective/GEFSv12/reforecast/ | head
aws s3 ls --no-sign-request s3://noaa-gefs-retrospective/GEFSv12/reforecast/2010/ | head -5
D=$(aws s3 ls --no-sign-request s3://noaa-gefs-retrospective/GEFSv12/reforecast/2010/ | awk '{print $2}' | head -1)
aws s3 ls --no-sign-request s3://noaa-gefs-retrospective/GEFSv12/reforecast/2010/$D | head -5
aws s3 ls --no-sign-request --recursive s3://noaa-gefs-retrospective/GEFSv12/reforecast/2010/$D | grep -i tmp_2m | head -10
hr "GEFS operational (noaa-gefs-pds)"
aws s3 ls --no-sign-request s3://noaa-gefs-pds/ | head -5; aws s3 ls --no-sign-request s3://noaa-gefs-pds/ | wc -l
aws s3 ls --no-sign-request s3://noaa-gefs-pds/gefs.20210105/00/ | head
aws s3 ls --no-sign-request s3://noaa-gefs-pds/gefs.20210105/00/atmos/ | head
aws s3 ls --no-sign-request s3://noaa-gefs-pds/gefs.20210105/00/atmos/pgrb2sp25/ | head -8
hr "Open-Meteo archive (realised, 2000)"
curl -s -m 60 "https://archive-api.open-meteo.com/v1/archive?latitude=41.88&longitude=-87.63&start_date=2000-01-01&end_date=2000-01-03&daily=temperature_2m_mean&timezone=America/Chicago" | head -c 400; echo
hr "Open-Meteo historical forecast API earliest"
for d in 2016-01-05 2018-01-05 2021-01-05 2022-01-05; do echo "-- $d"; curl -s -m 60 "https://historical-forecast-api.open-meteo.com/v1/forecast?latitude=41.88&longitude=-87.63&start_date=$d&end_date=$d&daily=temperature_2m_mean&models=gfs_seamless" | head -c 300; echo; done
hr "Open-Meteo previous runs API"
curl -s -m 60 "https://previous-runs-api.open-meteo.com/v1/forecast?latitude=41.88&longitude=-87.63&start_date=2022-01-05&end_date=2022-01-06&hourly=temperature_2m,temperature_2m_previous_day7&models=gfs_seamless" | head -c 500; echo
hr "Open-Meteo ensemble / single runs"
curl -s -m 60 "https://single-runs-api.open-meteo.com/v1/forecast?latitude=41.88&longitude=-87.63&run=2024-01-05T00:00&forecast_days=16&daily=temperature_2m_mean&models=gfs_seamless" | head -c 500; echo
hr "EIA storage history xls"
curl -s -m 60 -o ngshistory.xls -w "%{http_code} %{size_download}\n" https://ir.eia.gov/ngs/ngshistory.xls; file ngshistory.xls
curl -s -m 60 -o wngsr.xls -w "%{http_code} %{size_download}\n" https://ir.eia.gov/ngs/wngsr.xls
hr "EIA bulk NG zip"
curl -s -m 120 -I https://api.eia.gov/bulk/NG.zip | head -5
hr "Yahoo futures (yfinance)"
pip -q install yfinance >/dev/null 2>&1
python - <<'PY'
import yfinance as yf
for t in ["NG=F","NGH27.NYM","NGJ27.NYM","NGF27.NYM"]:
    try:
        d=yf.download(t,period="10d",progress=False,auto_adjust=False)
        print(t,len(d), d.tail(2).to_string()[:300])
    except Exception as e: print(t,"ERR",e)
PY
hr "CME settlements JSON"
curl -s -m 60 -A "Mozilla/5.0" -o cme.json -w "%{http_code} %{size_download}\n" "https://www.cmegroup.com/CmeWS/mvc/Settlements/Futures/Settlements/444/FUT?tradeDate=11/20/2026"; head -c 300 cme.json; echo
curl -s -m 60 -A "Mozilla/5.0" -o cmeo.json -w "%{http_code} %{size_download}\n" "https://www.cmegroup.com/CmeWS/mvc/Settlements/Options/Settlements/1352/OOF?tradeDate=11/20/2026"; head -c 300 cmeo.json; echo
hr "done"
