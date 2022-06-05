
function validateWelcomeEmail(email) {
    var re = /^(([^<>()[\]\\.,;:\s@\"]+(\.[^<>()[\]\\.,;:\s@\"]+)*)|(\".+\"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
    return re.test(email);
}

function checkwelcomesubscription(t,b){
	var email_address = document.getElementById(t).value;
	var email_signup = document.getElementById(b).value;
	
	if(validateWelcomeEmail(email_address)){
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
				set_fade_timer();
			}
		}
		
		xmlhttp_subscribe.open("POST","/Shop/EmailNotification/Confirmation.php",true);
		xmlhttp_subscribe.setRequestHeader("Content-type","application/x-www-form-urlencoded");
		xmlhttp_subscribe.send("email_signup="+email_signup+"&address="+email_address);
		jQuery.fancybox.close();
		enable_subscribe_pop();
	}else{
		window.location = ajaxFormatUrl("Shop/EmailNotification/Signup");
	}
}
