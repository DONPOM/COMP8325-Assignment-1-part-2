
if(document.cookie.indexOf("landing=")==-1) {
        var exdate=new Date();
        exdate.setDate(exdate.getDate()+365);
        document.cookie="landing=" +escape(document.location)+"|||"+escape(document.referrer)+";expires="+exdate.toGMTString()+";path=/;domain=.statcounter.com";
}
var sc_project=204609;
var sc_invisible=1;
var sc_security="0c932f53";
var sc_https=1;
var sc_click_stat=1;
var scJsHost = (("https:" == document.location.protocol) ? "https://secure." : "http://www.");
document.write("<sc"+"ript type='text/javascript' src='" + scJsHost + "statcounter.com/counter/counter_test4.js'></"+"script>");