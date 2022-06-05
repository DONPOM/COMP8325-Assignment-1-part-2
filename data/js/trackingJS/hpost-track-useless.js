
var _gaq = _gaq || [];
HPTrack.async = true;
HPTrack.getTracker('UA-71081-1');
HPTrack.setCustomVar("page", "front");
HPTrack.setCustomVar("vertical", "Homepage");
           
HPTrack.trackPageview();
    
(function() {
  var ga = document.createElement('script'); ga.type = 'text/javascript'; ga.async = true;
  ga.src = ('https:' == document.location.protocol ? 'https://ssl' : 'http://www') + '.google-analytics.com/ga.js';
  (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(ga);
})();

bN_cfg = {
  h: location.hostname,
  p: {"dL_ch":"us.hpmg","dL_flid":"_","dL_dpt":"front","cobrand":"HuffPost"}
};


function runOmni() {
 
 if(typeof HuffPoUtil != "undefined"){
  if (HuffPoUtil.getCookie('huffpost_user_id') != '')
    s_265.eVar17 = 'huffpost_' + HuffPoUtil.getCookie('huffpost_user_id');
  }

  s_265.prop54 = 'huffpost';
  s_265.pfxID = 'hpo';
  s_265.channel = 'us.hpmg';
  s_265.linkInternalFilters = 'javascript:,huffingtonpost.com';
  s_265.prop16 = 'page';
  s_265.prop1 = 'front';
  s_265.pageName = "" + document.title;
  s_265.prop12 = "" + document.URL.split('?')[0];
  s_265.t();
}

s_265_account ="aolhuffpo,aolsvc";

(function(d){
var s = d.createElement('script');
s.src = "http://o.aolcdn.com/os_merge/?file=/aol/beacon.min.js&file=/aol/omniture.min.js";
d.getElementsByTagName('head')[0].appendChild(s);
})(document);

