
//<![CDATA[
var smtId = "ab5bb963d";
var smtDefaultStyles = false;
var smtRedirect = true;
var smtProt = (("https:" == document.location.protocol) ? "https://" : "http://");

var smtPreRender = function(data) {
  for (i in data) {
    if (data[i].code === "en-us") {
      data[i].name = "English";
    }
  }
};

var smtRedirectMapper = function(locale, sites) {
  if (/^es/i.test(locale)){ //disable redirect to spanish site
    return null;
  }
  if (locale in sites) {
    return sites[locale];
  }
  if (/^fr/i.test(locale)) {
    return sites['fr-fr'] || null;
  }
  if (/^de/i.test(locale)) {
    return sites['de-de'] || null;
  }
  return null;
};

var smtElmt = document.createElement('script');
smtElmt.type = "text/javascript";
smtElmt.async = true;
smtElmt.src = smtProt + "cdn01.smartling.com/ls/"+ smtId +".js";
script = document.getElementsByTagName("script")[0];
script.parentNode.insertBefore(smtElmt, script);

//]]>
