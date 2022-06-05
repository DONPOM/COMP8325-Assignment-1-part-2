
var _comscore = _comscore || [];
(function() {
 var rp_cats = "IAB19";
 var rp_cat = 24*1000;
 if (rp_cats) {
	rp_cats = rp_cats.replace(/IAB/g,"").split(",");
	if (rp_cats.length > 0 && rp_cats[0] && rp_cats[0].length > 0) {
		rp_cat = rp_cats[0].split("-")[0];
		rp_cat = rp_cat * 1000;
	}
 }
 var _comscore_switch = "au";
 _comscore.push({ c1: "8", c2: "6135404", c3: rp_cat, c4: "12135", c10: "3423799" }); 
 if ( _comscore_switch == "us" )  {
   (function() { var s = document.createElement("script"), el = document.getElementsByTagName("script")[0]; s.async = true; s.src = (document.location.protocol == "https:" ? "https://sb" : "http://b") + ".scorecardresearch.com/beacon.js"; 
   el.parentNode.insertBefore(s, el); 
   })(); 
 }
   
})();
