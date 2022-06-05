var cnnDocDomain = "";
if (location.hostname.indexOf("cnn.com") > 0)
cnnDocDomain = "cnn.com";
if (location.hostname.indexOf("turner.com") > 0)
cnnDocDomain = "turner.com";
if (cnnDocDomain)
document.domain = cnnDocDomain;				