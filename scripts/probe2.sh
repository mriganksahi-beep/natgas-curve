#!/usr/bin/env bash
set +e
hr(){ echo; echo "=================== $1"; }
S="aws s3 ls --no-sign-request"
hr "reforecast date range"; $S s3://noaa-gefs-retrospective/GEFSv12/reforecast/ > rf.txt; head -3 rf.txt; tail -3 rf.txt
$S s3://noaa-gefs-retrospective/GEFSv12/reforecast/2019/ > rf19.txt; tail -3 rf19.txt; wc -l rf19.txt
$S s3://noaa-gefs-retrospective/GEFSv12/reforecast/2005/ > rf05.txt; head -3 rf05.txt; wc -l rf05.txt
$S "s3://noaa-gefs-retrospective/GEFSv12/reforecast/2005/2005011000/" ; $S "s3://noaa-gefs-retrospective/GEFSv12/reforecast/2005/2005011000/c00/"
curl -s "https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast/2005/2005011000/c00/Days%3A1-10/tmp_2m_2005011000_c00.grib2.idx" | head -5
curl -s "https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast/2005/2005011000/c00/Days%3A10-16/tmp_2m_2005011000_c00.grib2.idx" | head -3
hr "operational v11 2020-01"; $S s3://noaa-gefs-pds/gefs.20200115/00/ ; $S s3://noaa-gefs-pds/gefs.20200115/00/pgrb2a/ | grep gec00 | head -5; $S s3://noaa-gefs-pds/gefs.20200115/00/pgrb2a/ | grep gec00 | tail -3
$S s3://noaa-gefs-pds/gefs.20200115/00/pgrb2/ 2>/dev/null| grep gec00 | head -3
hr "operational v12"; $S s3://noaa-gefs-pds/gefs.20201001/00/atmos/pgrb2sp25/ | grep gec00 | tail -3
$S s3://noaa-gefs-pds/gefs.20211201/00/atmos/pgrb2ap5/ | grep gec00 | grep -v idx | sed -n '1,3p;$p'
$S s3://noaa-gefs-pds/gefs.20261120/00/atmos/pgrb2sp25/ | grep gec00 | grep -v idx | sed -n '1,2p;$p'
$S s3://noaa-gefs-pds/gefs.20261120/00/atmos/pgrb2ap5/ | grep gec00 | grep -v idx | sed -n '1,2p;$p'
curl -s https://noaa-gefs-pds.s3.amazonaws.com/gefs.20261120/00/atmos/pgrb2sp25/gec00.t00z.pgrb2s.0p25.f024.idx | grep -n TMP
curl -s https://noaa-gefs-pds.s3.amazonaws.com/gefs.20261120/00/atmos/pgrb2ap5/gec00.t00z.pgrb2a.0p50.f300.idx | grep -n "TMP:2 m"
curl -s https://noaa-gefs-pds.s3.amazonaws.com/gefs.20200115/00/pgrb2a/gec00.t00z.pgrb2af024.idx | grep -n "TMP:2 m"
curl -s https://noaa-gefs-pds.s3.amazonaws.com/gefs.20200115/00/pgrb2a/gec00.t00z.pgrb2af300.idx | grep -n "TMP:2 m"
hr "which day folders missing 2017-2026"; $S s3://noaa-gefs-pds/ | awk '{print $2}' | sed 's/gefs.//;s/\///' > days.txt; head -1 days.txt; tail -1 days.txt; wc -l days.txt
hr "CPC shapefile contents"
curl -s -o a.zip https://ftp.cpc.ncep.noaa.gov/GIS/us_tempprcpfcst/610temp_20130115.zip; unzip -l a.zip
curl -s -o b.zip https://ftp.cpc.ncep.noaa.gov/GIS/us_tempprcpfcst/814temp_20261120.zip; unzip -l b.zip
pip -q install pyshp >/dev/null 2>&1
python - <<'PY'
import shapefile, zipfile, io
for z in ["a.zip","b.zip"]:
    zf=zipfile.ZipFile(z); names=zf.namelist(); base=[n[:-4] for n in names if n.endswith(".shp")][0]
    r=shapefile.Reader(shp=io.BytesIO(zf.read(base+".shp")),dbf=io.BytesIO(zf.read(base+".dbf")))
    print(z, r.fields); print(r.records()[:6]); print(len(r.shapes()), r.bbox)
PY
hr "EIA history"
curl -sL -o h.xls -w "%{http_code} %{size_download} %{url_effective}\n" https://ir.eia.gov/ngs/ngshistory.xls; file h.xls
curl -sL -o h2.xls -w "%{http_code} %{size_download} %{url_effective}\n" https://www.eia.gov/naturalgas/storage/ngshistory.xls; file h2.xls
curl -sL -o bulk.zip -w "%{http_code} %{size_download}\n" https://www.eia.gov/opendata/bulk/NG.zip; unzip -p bulk.zip | grep -o '"series_id":"NG.N[^"]*SWO[^"]*W"' | sort -u | head -40
unzip -p bulk.zip | grep '"series_id":"NG.NW2_EPG0_SWO_R48_BCF.W"' | grep -o '"start":"[0-9]*"'
unzip -p bulk.zip | grep -E '"series_id":"NG.(RNGC1|RNGC4|RNGWHHD).D"' | grep -oE '"(series_id|start|end)":"[^"]*"'
hr done
