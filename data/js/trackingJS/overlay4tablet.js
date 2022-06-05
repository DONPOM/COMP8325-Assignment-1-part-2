//function FC2AdTabletOverlay(option) {
function FC2AdTabletOverlay(adname) {
	//var url = option.url, src = option.src, offsetHeight;
	var offsetHeight;
	//window.addEventListener('load', function() {
		var getZoom = function() { return Math.round(window.innerWidth / document.documentElement.clientWidth * 100)/100; };
		var getScrollMax = function() { return document.documentElement.getBoundingClientRect().height - window.innerHeight; }

		var ad = document.getElementById(adname);

		var show = function() {
			var showTime = setTimeout(function() {
			clearTimeout(showTime);
			ad.style.display = 'block'
			}, 200);
		};

		var hide = function() {
			ad.style.display = 'none';
			
		}

		var scrollHandler = function(e) {
			try{
			var zoom = getZoom();
			ad.style.webkitTransform = 'scale('+zoom+')';
			ad.style.left = window.pageXOffset + 'px';
			var setting = getScrollMax() * 0.9 < window.pageYOffset? ['top', window.pageYOffset] : ['bottom', window.pageYOffset + window.innerHeight - offsetHeight];
			ad.style.webkitTransformOrigin = 'left '+setting[0];
			ad.style.top = setting[1] + 'px';
			show();
			}catch(er){
			
			}
		};

		/*
		var imgLoadHandler = function() {
			this.onload = null;
			offsetHeight = ad.offsetHeight;
			hide();
			ad.style.visibility = 'visible';
			
			if(window.addEventListener){
				window.addEventListener('scroll', scrollHandler, false);
				window.addEventListener('resize', scrollHandler, false);
			}
			if(document.addEventListener){
				document.addEventListener('touchmove', hide, false);
				document.addEventListener('touchend', show, false);
			}
			setTimeout(scrollHandler, 1000);
		} //*/

		//document.body.appendChild(ad);
		ad.style.position = 'absolute';
		ad.style.textAlign = 'center';
		ad.style.visibility = 'hidden';
		ad.style.top = 0;
		ad.style.width = '100%';
		
		
		//append
		offsetHeight = ad.offsetHeight;
		hide();
		ad.style.visibility = 'visible';
		if(window.addEventListener){
	 		window.addEventListener('scroll', scrollHandler, false);
	 		window.addEventListener('resize', scrollHandler, false);
		}
		if(document.addEventListener){
	 		document.addEventListener('touchmove', hide, false);
	 		document.addEventListener('touchend', show, false);
	 	}
	 	setTimeout(scrollHandler, 1000);		
		
		
		//ad.innerHTML = '<a href="'+url+'" target="_blank"></a>';
		//ad.innerHTML = html;

		//var adImg = new Image();
		//adImg.onload = imgLoadHandler;
		//adImg.src = src;
		//ad.firstChild.appendChild(adImg);
	//}, false);
}
