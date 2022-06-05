new function(){
	var CNAME = 'fclo';
	var COOKIE_DOMAIN = 'id.fc2.com';
	var ua = navigator.userAgent;
	var d = new Date();
	var lang = (navigator.language) ? navigator.language : navigator.userLanguage;

	var now = new Date();
	var ja1 = new Date(now.getFullYear(), 0, 1, 0, 0, 0, 0);
	var tmp = ja1.toGMTString();
	var ja2 = new Date(tmp.substring(0, tmp.lastIndexOf(" ")-1));
	var std_time_offset = (ja1 - ja2) / (1000 * 60 * 60);

	var ju1 = new Date(now.getFullYear(), 6, 1, 0, 0, 0, 0);
	tmp = ju1.toGMTString();
	var ju2 = new Date(tmp.substring(0, tmp.lastIndexOf(" ")-1));
	var daylight_time_offset = (ju1 - ju2) / (1000 * 60 * 60);
	var dst = '';
	if (std_time_offset != daylight_time_offset) {
		dst = ' DST';
	}
	var nowtime = new Date().getTime();
	var clear_time = new Date(nowtime + (60 * 60 * 24 * 1000 * 365));
	var expires = clear_time.toGMTString();

	function getCookieDomain(){
		var host = location.host;
		var m = host.match(/([a-z0-9][a-z0-9\-]{1,63}\.[a-z]{2,6})$/);
		if(m && m.length > 0){
			return 'id.' + m[0];
		}
		return COOKIE_DOMAIN;
	}
	var domain =  getCookieDomain();
	var cval = CNAME + '=' + escape(now.getTime()  + ',' + lang + ',' + std_time_offset + dst  ) + ';domain=.' + domain  + ';path=/;expires=' + expires;
	document.cookie = cval;
};
