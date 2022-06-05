var MaskDialog=new maskDialog(400,303);
window.onload=function(){
		document.portalForm.reset();
		MaskDialog.mainForm=document.getElementById("div_Dialog");
		};
//显示蒙板
function selectProvince(type) { //找回备案密码
	if(type == '0') {
		document.getElementById("atype").value='0';
	} else if(type == '1') { //修改备案密码
		document.getElementById("atype").value='1';
	} else if(type == '2') { //备案证书下载
		document.getElementById("atype").value='2';
	}
	                    	
	MaskDialog.show();
}
//选定省份并提交
function provinceSelected() {
	var atype = document.getElementById("atype").value;
	var subCenterId ="";
	var radios = document.getElementsByName("toSubCenter");
	var flag = false;
	for(var i=0;i<radios.length;i++) {
		if(radios[i].checked == true) {
			subCenterId = radios[i].value;
			flag = true;
			break;
		}
	}
	if(!flag) return;
	var url="";
 	if(atype == '0') {
 		url = '/state/outPortal/toGetPassword.action?atype=0';
 	} else if(atype=='1') {
		url = '/state/outPortal/toModifyPassword.action?atype=1';
 	} else if(atype=='2') {
 		url = '/state/outPortal/toCertificateDownload.action?atype=2';
 	}
 	if(url!="") {
 		url+="&subCenterId="+subCenterId;
 	}
 	window.open(url,"toPage","directories=yes,height=600,width=980,location=yes,menubar=yes,resizable=yes,scrollbars=yes,status=yes,titlebar=yes,toolbar=yes,left=120,top=80",false);
 	MaskDialog.hide();
 	for(var i=0;i<radios.length;i++) {
		if(radios[i].checked == true) {
			radios[i].checked=false;
			break;
		}
	}
}

//管理用户点击“进入”
function signIn() {
	var opts = document.getElementById("urlProvinceSel").options;
	var selectedVal = "";
	var url = "/state/outPortal/toManageUserSystem.action";
	for(var i=0;i<opts.length;i++) {
		if(opts[i].selected == true) {
			selectedVal = opts[i].value;
			break;
		}
	}
	if(selectedVal == '-1') return;

	url+="?subCenterId=" + selectedVal;
	window.open(url,"_RMS_","directories=yes,height=600,width=980,location=yes,menubar=yes,resizable=yes,scrollbars=yes,status=yes,titlebar=yes,toolbar=yes,left=120,top=80",false);
}

function reloadImage(url){
	   document.portalForm.img1.src = url+"?dumy="+Math.random(); 
} 

function queryISP(arg){//查询报备单位
	var myForm =document.portalForm;
	document.getElementById('error_message').innerHTML="";
	var verifyCode = document.getElementById('verifyCode').value;
	if(checkKeyword('',false)){
		if(verifyCode==''){
			alert("验证码不能为空。");
			return false;
		}
	}else{
		 return false;
	}
	var message;
	$.ajax({
	   type: "POST",
	   url: "/common/validate/validateVerifyCode.action",
	   data: "validateValue="+encodeURI(encodeURI(verifyCode)),
	   dataType: "json",
	   success: function(data){
		   	if(data.result == true) {
		   		var unitStr = document.getElementById('unitStr').value;
				var province = document.getElementById('province').value;
				var verifyCode = document.getElementById('verifyCode').value;
				document.portalForm.action = arg;
				document.portalForm.target="_blank";
				document.portalForm.submit();
				document.portalForm.province.value = 0;
				document.portalForm.unitStr.value = '';
				document.portalForm.verifyCode.value = '';
				changeCaptcha();
		   	}else {
		   		alert("输入验证码不正确，请重新输入。");
		   		return false;
		   	}
	   }
	}); 
	return true;
}