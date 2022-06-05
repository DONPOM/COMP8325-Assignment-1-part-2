try{
	document.domain="qq.com";
}catch(e){}
function ptlogin2_onResize(width, height) {
	login_wnd = document.getElementById("login_div");
	if (login_wnd)  {
		login_wnd.style.width = width + "px";
		login_wnd.style.height = height + "px";
		login_wnd.style.visibility = "hidden"
		login_wnd.style.visibility = "visible"
		login_wnd.style.marginLeft = "-"+parseInt(width/2) + 'px';
		login_wnd.style.marginTop = "-"+parseInt(height/2) + 'px';
	}
}
function ptlogin2_onClose(){
	document.getElementById("login_div").style.left = "-9999px";
	document.getElementById("layoutBg").style.display = "none";
}
function userLogin(){
	document.getElementById("layoutBg").style.display = "block";
	document.getElementById("layoutBg").style.height = document.body.clientHeight + "px";
	document.getElementById("login_frame").src = "http://ui.ptlogin2.qq.com/cgi-bin/login?hide_title_bar=0&low_login=0&qlogin_auto_login=1&no_verifyimg=1&link_target=blank&appid=636014201&target=self&s_url=http%3A//www.qq.com/qq2012/loginSuccess.htm";
	document.getElementById("login_div").style.left = "50%";
}

