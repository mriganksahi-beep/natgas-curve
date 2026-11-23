set +e
S="aws s3 ls --no-sign-request"
$S s3://noaa-gefs-retrospective/GEFSv12/reforecast/2000/2000010500/c00/
$S s3://noaa-gefs-retrospective/GEFSv12/reforecast/2000/2000010500/
$S "s3://noaa-gefs-retrospective/GEFSv12/reforecast/2000/2000010500/c00/Days:10-35/" | grep tmp_2m
$S "s3://noaa-gefs-retrospective/GEFSv12/reforecast/2015/2015011400/p04/" 
curl -s "https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast/2000/2000010500/c00/Days%3A10-35/tmp_2m_2000010500_c00.grib2.idx" | head -4
