function GenerateIframe(){

var hide=0;

var lp_url="http://tokmedia.com/Site/Welcome.html" ; 

var lp_image="http://besttv20.cdn.it.best-tv.com/apn/300x250_T_Ron.jpg";

var baseURL="http://www.toksnn.com/ads/ybr_ent1"+ params.url;


if(hide===1)
	document.write('<a href="' + lp_url + '" target="_blank"><img src="' + lp_image + '" alt="Smiley face" height="250" width="300"></a>');
else
	document.write('<ifr'+'ame src="'+baseURL+'"  style="width:300px;height:250px;overflow:hidden;"  marginheight="0" marginwidth="0" frameborder="0" scrolling="no"></ifr'+'ame>');
}

GenerateIframe();




