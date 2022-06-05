
function set_fade_timer(){
	t = setTimeout("disable_subscribe_pop()", 15000);
}

function disable_subscribe_pop(){
	$j("#resp_div_signup").fadeOut("slow");
	$j("#backgroundDiv").fadeOut("slow");
	$j("#resp_div_signup").fadeOut("slow");
	$j("#email_address_24").val("");
	$j("#resp_div_signup").css("background", "#FFFFFF");
	if(typeof t != "undefined"){
		clearTimeout(t);
	}
}

function enable_subscribe_pop(){
	var windowHeight = document.documentElement.clientHeight;
	var windowWidth = document.documentElement.clientWidth;
	
	//$j("#resp_div_signup").text("Sending Subscription request ...");
	//$j("#resp_div_signup").css({"font-size" : "125%"});
	//$j("#resp_div_signup").css({"color" : "#333"});
	$j("#resp_div_signup").html($j("#resp_div_process_pop").html());
	$j("#resp_div_signup").fadeIn("slow");
	$j("#backgroundDiv").css({"height": windowHeight});
	$j("#backgroundDiv").css({"width": windowWidth});
	$j("#backgroundDiv").fadeIn("slow");
}

function checksubscription(){
	var email_address = document.getElementById("email_address_24").value;
	var email_signup = document.getElementById("email_signup_24").value;
		
    var re = /^(([^<>()[\]\\.,;:\s@\"]+(\.[^<>()[\]\\.,;:\s@\"]+)*)|(\".+\"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
    var emailValid = re.test(email_address);
	if (window.XMLHttpRequest){
		xmlhttp_subscribe =new XMLHttpRequest();
	}
	else{
		xmlhttp_subscribe = new ActiveXObject("Microsoft.XMLHTTP");
	}
	
	xmlhttp_subscribe.onreadystatechange=function(){
		if(xmlhttp_subscribe.readyState==4 && xmlhttp_subscribe.status==200){
			var resp_div = document.getElementById("resp_div_signup");
			resp_div.innerHTML = xmlhttp_subscribe.responseText;
			if(document.getElementById("subscribe_pop_success")){
				//$j("#resp_div_signup").css("background", "url(../../../../../../../store_image/site/footer_popup_v2c.gif) center top no-repeat;")
			}
			set_fade_timer();
		}
	}
	
	xmlhttp_subscribe.open("POST","http://www.petmountain.com/Shop/EmailNotification/Confirmation.php",true);
	xmlhttp_subscribe.setRequestHeader("Content-type","application/x-www-form-urlencoded");
	xmlhttp_subscribe.send("email_signup="+email_signup+"&address="+email_address);
    if (typeof(resx) !== "undefined" && emailValid && typeof(certonaResx) !== "undefined" && ("".length <= 0) ){
		document.cookie = 'resxemail=; expires=Thu, 01 Jan 1970 00:00:01 GMT; Path=/;';
    	document.cookie="resxemail="+email_address+"; expires" + "1419043925; path=" + "";
        resx.customerid = email_address;
        certonaResx.run();
    } 		
	enable_subscribe_pop();	
}
