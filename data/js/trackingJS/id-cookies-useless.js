
	if (window.location.href.indexOf("fromApp") != -1) {
		
		createCookie("legalaidapp",1);
		
	}	
	var hasCookie=document.cookie.indexOf("legalaidapp") > -1;	
	
	var ua = navigator.userAgent.toLowerCase();
	
	var isAndroid = ua.indexOf("android") > -1; //&& ua.indexOf("mobile");
	//var isIPhone = ua.indexOf("iphone") > -1;
	var isIPAD = ua.indexOf("ipad") > -1;
	var isIPOD = ua.indexOf("ipod") > -1;
	var isIPhone=ua.indexOf("iphone") > -1;
	var isiOS6=ua.indexOf("os 6_0") > -1;
	

if (!hasCookie) {
	//ask user to download if on mobile device
	if (isIPhone || isIPhone || isIPAD ) {
		if (isOS6) {
			
		}
		else {
			window.location = 'http://www.illinoislegalaid.org/downloadLegalAidApp.cfm?type=iphone&qString=';
		}
	}
	if (isAndroid ) {
	//	window.location = 'http://www.illinoislegalaid.org/downloadLegalAidApp.cfm?type=android&qString=';
	}
	
}