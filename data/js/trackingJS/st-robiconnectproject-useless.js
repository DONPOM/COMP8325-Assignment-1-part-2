
var rHost = (("https:" == document.location.protocol) ? "https" : "http");
var sd = ""; var gkw = ""; var gch = ""; var rkw = "";
gkw += "google_kw=ecommerce";
gch += "google_ad_channel=9186617663+7252437058";
rkw += "keyword=" + gch.replace(/google_ad_channel=/, '');
sd += "&" + gkw + "&" + gch + "&" + rkw;
var cb = Math.random();
var d = document;
d.write('<script type="text/javascript"');
d.write('src="'+rHost+'://optimized-by.rubiconproject.com/a/7554/12135/20985-9.js?cb='+cb+sd+'">');
d.write('<\/scr'+'ipt>');
