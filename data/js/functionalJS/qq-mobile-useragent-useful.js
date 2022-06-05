
if(window.location.toString().indexOf('pref=padindex') != -1){
}else{
	if(/AppleWebKit.*Mobile/i.test(navigator.userAgent) || (/MIDP|SymbianOS|NOKIA|SAMSUNG|LG|NEC|TCL|Alcatel|BIRD|DBTEL|Dopod|PHILIPS|HAIER|LENOVO|MOT-|Nokia|SonyEricsson|SIE-|Amoi|ZTE/.test(navigator.userAgent))){  
      if(window.location.href.indexOf("?mobile")<0){
		try{
			if(/Android|Windows Phone|webOS|iPhone|iPod|BlackBerry/i.test(navigator.userAgent)){
				window.location.href="http://xw.qq.com/index.htm";
			}else if(/iPad/i.test(navigator.userAgent)){
              	window.location.href="http://www.qq.com/pad/"
			}else{
				window.location.href="http://xw.qq.com/simple/s/index/"
			}
		}catch(e){}
	}
	}
}
