
		(function () {
			var enableFarmTimingSampling = true,
				enablePageTimingSampling = true,
				enableLightHouseTimingSampling = false;

		
		function lighthouseBeacon(){if(window.LH&&window.LH.isInitialized){LH.tag('pageType',{val:'sihp'});if(F.config&&F.config.flickr.user.user_ok){LH.tag('l');} LH.beacon();}else{setTimeout(function(){lighthouseBeacon();},500);}} var c=document.getElementById('firstCard'),it=c.getElementsByTagName('img')[0];it.onload=function(){if(typeof page_timing!=='undefined'){page_timing.time_to_first_photo=new Date().getTime();} if(enableLightHouseTimingSampling&&window.LH){LH.mark('mark_time_to_first_photo');lighthouseBeacon();}}})();
		
	