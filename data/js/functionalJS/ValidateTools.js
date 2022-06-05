/**
 * 公共校验方法js
 * 
 * 依赖文件：/js/common.js
 * 
 * 
 * */



	//非空验证
	
	function checkNull(str){
		
		if(str!=null&&str!=''){
			return true;
		}
		return false;
	}
	
	
	//手机号验证
	
	function checkMobile(str){
		
		if(str==null||str==''){
			return false;
		} 

		return (/^0?(130|131|132|153|156|185|186|188|134|135|136|137|138|139|157|158|159|133|153|180|189)[0-9]{8}/.test(str));
	}
	
	//座机号验证
	
	function checkTelephone(str){
		if(str==null||str==''){
			return false;
		} 

		var exp1 = /^[0-9\-\;]*$/;
		var exp2 = /^(086-(0\d{3}|0\d{2})-(\d{8}|\d{7}))+(-(\d{4}|\d{3}))?/;
		
		if(!(exp1.test(str) && exp2.test(str))){
			return false;
		}

		return true;
	}
	
	//身份证验证

	function checkPersonId(str){
		if(str==null||str==''){
			return false;
		}
		
		var exp1=/(^\d{15}$)|(^\d{17}([0-9]|X)$)/;

		if(exp1.test(str)==false){
		
			return false;
		
		}else if(str.length==18){
		
			var iSs = 0;
			
			var iY=0;
		
			var szVerCode="10X98765432";
		
			var iw=new Array();
		
			iw[0] = 7;
			iw[1] = 9;
			iw[2] = 10;
			iw[3] = 5;
			iw[4] = 8;
			iw[5] = 4;
			iw[6] = 2;
			iw[7] = 1;
			iw[8] = 6;
			iw[9] = 3;
			iw[10] = 7;
			iw[11] = 9;
			iw[12] = 10;
			iw[13] = 5;
			iw[14] = 8;
			iw[15] = 4;
			iw[16] = 2;	
			
			for(i=0;i<17;i++){
			
				iSs+= iw[i]*(str.substr(i,1)-"0");
			
			}
						
			iY = iSs % 11;
			
			var eChr=str.substr(17,1);
			
			var vChr=szVerCode.substr(iY,1);
						
			if(eChr==vChr){
			
				return true;
			
			}else{
			
				return false;
			
			}
			
			
		
		}else{
		
			//15位 未增加算法验证			
		
			return true;
		
		}

		
	}

//非法字符验证
	
	function checkIllegalChar(str){
		var strLow=str.toLowerCase();

		var non_special_char_regexp = " ; ,> , <, - , * , % , ^ , + , $ , ' ,copy ,format , char(, and , or , exec ,insert ,select ,delete ,update ,count(, chr , mid , master ,truncate , declare ,div ,shutdown,drop ,alter , like ,html ,input ,href ,iframe ,onclick ,script,alert, tolower,procedure ,where , codec , from , user , table ";
		
		var a = non_special_char_regexp.split(",");
		
		var strint = strLow.length;
		
		for(i=0; i<a.length; i++){
			
			strLow=strLow.replace(a[i], "");
			if(strLow.length<strint){
				return false;
			}
			
		}
		
		return true;
	}
	
	/*
	 * 校验字符长度
	 * @param word
	 * @param len
	 * @return true/false
	 */
	function checkWordLength(word, len){
		if(word==null) return true;
		var lth = word.replace(/[^\x00-\xFF]/g,'**').length;
		if(lth>len) {
			return false;
		} 
		return true;
	}
	
	/*验证所有文本输入信息是否合法  keywords 关键字集合  cell  true:验证关键字，非法字符 false:验证非法字符
	*返回结果：true/false
	*/
	function checkKeyword(keywords,cell){
		var checkFlag = "true";
 		$(":text").each(function(i){
 			//判断是否包含非法字符和脏字
 			if(checkFlag == "true"){
 				var inputWidth = this.style.width;
 				//清空样式
 				//this.style.cssText = "";
 				if(cell){//验证关键字和非法字符
		 			if(includeKeywords(keywords,this.value) || !checkIllegalChar(this.value)){
		 				alert("输入包含非法字符：\n"+this.value);
		 				this.style.cssText = "border:2px solid #FFCC88;";
		 				this.style.width = inputWidth ;
		 				//document.getElementById('errorDiv').style.display = 'block';
		 				//document.getElementById('errorDiv').innerHTML = "[ "+this.value + " ] 中包含非法字符";
		 				checkFlag =  "false";
		 			}
 				}else{//验证非法字符
 					
		 			if(!checkIllegalChar(this.value)){
		 				alert("输入包含非法字符：\n"+this.value);
		 				//this.style.cssText = "border:2px solid #FFCC88;";
		 				//this.style.width = inputWidth ;
		 				//document.getElementById('errorDiv').style.display = 'block';
		 				//document.getElementById('errorDiv').innerHTML = "[ "+this.value + " ] 中包含非法字符";
		 				checkFlag =  "false";
		 			}
 				}
 				if(checkFlag == "true"&this.value!=''){//如果不是非法字符，再校验字符的长度，因为要用div显示，所以放在验证非法字符后面
 					
 			 		if(bytes(this.value)>512){//判定输入的字符长度不能大于512，如果超过提示用户
 			 			this.style.cssText = "border:2px solid #FFCC88;";
 						this.style.width = inputWidth ;
 						document.getElementById('errorDiv').style.display = 'block';
 						document.getElementById('errorDiv').innerHTML = "输入的内容过长，范围512位字母、数字或汉字（汉字算两位）";
 						checkFlag =  "false";
 			 		}
 				}
 			}
 		});
 		if(checkFlag == "false"){
 			return false;
 		}
 		return true;
	}
	
	/*验证所有文本输入信息是否合法  keywords 关键字集合  cell  true:验证关键字，非法字符 false:验证非法字符
	 * @param 式json格式
	 * 
	 *返回结果：true/false
	 */
	function checkAllFieldKeywords(keywords,cell,p){
		//校验文本域text
		var flag1 = checkKeyword(keywords,cell);
		var flag2 = checkTextareaKeywords(keywords,cell,p);
		if(flag1 && flag2){
			return true;
		}else{
			return false;
		}
	}
	
	function checkTextareaKeywords(keywords,cell,p) {
		if(p==null || typeof(p)=='undefined') return false;
		var flag = true;
		var k = "";
		for(k in p) {
			var obj = document.getElementById(k);
			//if(p[k] == null || obj.tagName != 'textarea') continue;
			if(flag) {
				if(cell){//验证关键字和非法字符
		 			if(includeKeywords(keywords,obj.value) || !checkIllegalChar(obj.value)){
		 				var inputWidth = obj.style.width;
		 				alert("输入包含非法字符：\n"+obj.value);
		 				obj.style.cssText = "border:2px solid #FFCC88;";
		 				obj.style.width = inputWidth ;
		 				flag =  false;
		 			}
				}else {//验证非法字符
		 			if(!checkIllegalChar(obj.value)){
		 				alert("输入包含非法字符：\n"+obj.value);
		 				flag = false;
		 			}
				}
				
				if(bytes(obj.value) > parseInt(p[k])){//判定输入的字符长度不能大于参数限制
		 			obj.style.cssText = "border:2px solid #FFCC88;";
					obj.style.width = inputWidth ;
					document.getElementById('errorDiv').style.display = 'block';
					document.getElementById('errorDiv').innerHTML = "输入的内容过长，范围" + p[k] + "位字母、数字或汉字（汉字算两位）";
					flag= false;
		 		}
				
				if(!flag) {
					break;
				}
			}
			
		}
		return flag;
		
	}
	
	//得到字符串的字节长度
	 function bytes(str)
	 {
	 	if(typeof(str)!='string')
	 	{
	 		str = str.value;
	 	}
	 	var len = 0;
	 	for(var i = 0; i < str.length; i++){
	 		if(str.charCodeAt(i) > 127){
	 			len++;
	 		}
	 		len++;
	 	}
	 	return len;
	 }

	
	//关键字验证
	function includeKeywords(keywords,cell){
		var keys = new Array();
		var key = keywords;
		if(key.indexOf(";")>=0){
			keys = key.split(";");
		}else{
			keys[0]=key;
		}
		if(keywords == ""){
			return false;
		}
		for(var i=0;i<keys.length;i++){
			if(cell.indexOf(keys[i])>=0){
				return true;
			}
		}
		return false;
	}

	//数字验证
	
	function checkNumber(str){
		
		if(str==null||str==''){
			return false;
		}

		return /^[0-9]*[1-9][0-9]*$/.test(str);
	}
	
	//纯字母验证

	function checkChar(str){
		
		if(str==null||str==''){
			return false;
		}

		return /^[A-Za-z]*$/.test(str);
	}
	
	//电子邮件验证
	
	function checkEmail(str){
	
		if(str==null||str==''){
			return false;
		}

		var result=str.match(/^\w+((-\w+)|(\.\w+))*\@[A-Za-z0-9]+((\.|-)[A-Za-z0-9]+)*\.[A-Za-z0-9]+$/);   
	    if(result==null) return false;   
	    return true;
	
	}
	
	
	function checkForEmail(email){
		var xx = new RegExp("^\w+((-\w+)|(\.\w+))*\@[A-Za-z0-9]+((\.|-)[A-Za-z0-9]+)*\.[A-Za-z0-9]+$");
		if(!xx.test(email)){
			return false;
		}
		return true;
	}
	
	//日期验证
	
	function checkDate(str){
	
	
		if(str==null||str==''){
			return false;
		}

		return /([0-9]{3}[1-9]|[0-9]{2}[1-9][0-9]{1}|[0-9]{1}[1-9][0-9]{2}|[1-9][0-9]{3})-(((0[13578]|1[02])-(0[1-9]|[12][0-9]|3[01]))|((0[469]|11)-(0[1-9]|[12][0-9]|30))|(02-(0[1-9]|[1][0-9]|2[0-8])))/.test(str);
		
	}
	
	//验证域名
	
	function checkDomain(str){
	
		if(str==null||str==''){
			return false;
		}
		var strRegex = "^((https|http|ftp|rtsp|mms)?://)"  
			  + "?(([0-9a-z_!~*'().&=+$%-]+: )?[0-9a-z_!~*'().&=+$%-]+@)?" //ftp的user@  
			        + "(([0-9]{1,3}\.){3}[0-9]{1,3}" // IP形式的URL- 199.194.52.184  
			        + "|" // 允许IP和DOMAIN（域名） 
			        + "([0-9a-z_!~*'()-]+\.)*" // 域名- www.  
			        + "([0-9a-z][0-9a-z-]{0,61})?[0-9a-z]\." // 二级域名  
			        + "[a-z]{2,6})" // first level domain- .com or .museum  
			        + "(:[0-9]{1,4})?" // 端口- :80  
			        + "((/?)|" // a slash isn't required if there is no file name  
			        + "(/[0-9a-z_!~*'().;?:@&=+$,%#-]+)+/?)$";  
			        var re=new RegExp(strRegex);  
			  //re.test() 
			        if (re.test(str)){ 
			            return (true);  
			        }else{  
			            return (false);  
			        } 

		
	}
	
	//验证ip地址
	
	function checkIp(str){

		if(str==null||str==''){
			return false;
		}
		var pattern=/^(25[0-5]|2[0-4][0-9]|[0-1]{1}[0-9]{2}|[1-9]{1}[0-9]{1}|[1-9])\.(25[0-5]|2[0-4][0-9]|[0-1]{1}[0-9]{2}|[1-9]{1}[0-9]{1}|[1-9]|0)\.(25[0-5]|2[0-4][0-9]|[0-1]{1}[0-9]{2}|[1-9]{1}[0-9]{1}|[1-9]|0)\.(25[0-5]|2[0-4][0-9]|[0-1]{1}[0-9]{2}|[1-9]{1}[0-9]{1}|[0-9])$/;
		return pattern.test(str);

	}
	
	/*判断起始IP是否小于终于IP
	* 黄志鲁
	*/
  function compareIp(startIp,endIp){
		if (startIp == endIp) {
			return true;
		}
	    //分解每个域作判断
		for (var i = 0; i < 4; i++) {
			var index1 = startIp.indexOf(".");
			var index2 = endIp.indexOf(".");
			var tmp1;
			var tmp2;
			if (-1 != index1) {
				tmp1 = startIp.substring(0, index1);
				tmp2 = endIp.substring(0, index2);
				startIp = startIp.substring(index1 + 1, startIp.length);
				endIp = endIp.substring(index2 + 1, endIp.length);
			} else {
				tmp1 = startIp.substring(0, startIp.length);
				tmp2 = endIp.substring(0, endIp.length);
			}
			
			if (1 == compInteger(tmp1, tmp2)) {
				return false;
			}
			if (2 == compInteger(tmp1, tmp2)) {
				break;
			}
		}
		return true;

	}
		//比较整数大小
	//黄志鲁
	function compInteger(strValue1, strValue2){
	      var  nValue1, nValue2 ;
	      nValue1 = parseInt(strValue1) ;
	      nValue2 = parseInt(strValue2) ;
	
	      if (nValue1 > nValue2){
	          return 1 ;
	       }
	      if (nValue1 == nValue2){
	          return 0 ;
	       }
	      if (nValue1 < nValue2){
	          return 2 ;
	       }   
	  }
	     //验证字符串的长度，不大于指定的值
	//罗飞翔   
    function compStrLength(strValue,strLength){
       if(strValue.length>parseInt(strLength)){
          return false;
       }
        return true;
    }
   
	