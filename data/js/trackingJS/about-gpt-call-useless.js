



//Set mpt and GMCID cookies
setCookieValue();

Abt.GPT.initGPT({
	site: 'www',
	channel: 'homepage',
	articleId: '',
	templateId: '19',
	articleStamp: '',
	pageCount: 'null',
	hgt: '' + window.screen.height,
	isMobile: false,
	singleRequest: false,
	slots: [
		{'id':'billboard', 'pos':'atf', 'sizes':[[300, 250], [300, 600], [300, 1050]], 'type':'billboard', 'priority':'5'}
	]
});

// analytics/404.ftl

// analytics/googleTagManager.ftl

var dataLayer = dataLayer || [];
// AB Testing
dataLayer.push({
abTests: [{
testName:"relatedClickabilityTest",
bucketDescription:"mongo bucket",
bucketName:"mongo",
bucketTrackingId:"19",
bucketValue:2
},{
testName:"billboard4Test",
bucketDescription:"enabled",
bucketName:"enabled",
bucketTrackingId:"99",
bucketValue:1
},{
testName:"seoDateStampTest",
bucketDescription:"enabled",
bucketName:"enabled",
bucketTrackingId:"99",
bucketValue:1
}]
});
//Environment Tracking
var envData = {
environment : {
environment : "prod",
application : "globe",
dataCenter : "sj",
serverName : "sjglobe14"
},
server : {
version : "1.15.4",
vendor : "",
title : "Globe Server"
},
resources : {
version : "1.15.4",
loadStartTime : "1417642344472",
loadTimeTaken : "1604"
},
client : {
browserUA : navigator.userAgent,
serverUA : "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/600.1.17 (KHTML, like Gecko) Version/7.1 Safari/537.85.10"
}
};
dataLayer.push({ envData: envData });

if ((typeof opt_testid !== 'undefined') && (opt_testid !== '')){
	docCookies.setItem("opt_testid", opt_testid, null, "/", '.about.com');
} else if (docCookies.getItem('opt_testid')) {
	var opt_testid = docCookies.getItem('opt_testid'); 
} else {
	var opt_testid = ''; 
}


var ch = 'homepage',
	gs = 'www',
	zGAst = '',
	zAth = '',
		zTt = '19',
		zGAtg ='Homepage',
	google_ad_channel = 'primedia_channel_homepage tt' + zTt,
	google_ad_client = docCookies.getItem('mpt') == 'A1' ? 'ca-aj-about-sem' : 'ca-aj-about-premium';

var askoc = docCookies.getItem('askoc'),
	mptCookie = docCookies.getItem('mpt'); 
google_ad_channel += ((typeof askoc !== 'undefined' && askoc !== null) ? ' '+askoc+' semd' : '');
google_ad_channel += ((opt_testid !== 'undefined' && opt_testid !== null) ? (' test' + opt_testid) : '');
google_ad_channel += ((typeof mptCookie !== 'undefined' && mptCookie !== null) ? ' ' + mptCookie : '');

var ab_tracking_ids = [];
		ab_tracking_ids.push("19");
		ab_tracking_ids.push("99");
		ab_tracking_ids.push("99");
if (ab_tracking_ids.length) {
	ab_tracking_ids.forEach(function (id) {
		if (id !== 'undefined' && id !== null) {
			google_ad_channel += ' test' + id;
		}
	});
}

var google_ad_channel_window = google_ad_channel;

if(docCookies.keys().length > 0){
	var 
		zBsT  = docCookies.getItem('zBsT'),
		zBT   = docCookies.getItem('zBT'),
		zBTr  = docCookies.getItem('zBTr'),
		mpt   = docCookies.getItem('mpt');
}

var zNative;
	zNative = 0;