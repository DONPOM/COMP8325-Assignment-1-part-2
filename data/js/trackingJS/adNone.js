/*
@author   lingchen
@date:    2013-03-30
@edition: 1.0
@info:    浪首通栏广告隐藏机制（除顶通以外）
*/
(function(document,window){
document.domain = 'sina.com.cn';	
var getByClassName = function(className){  
	if(document.getElementByClassName){  
		return document.getElementByClassName(className) //标准浏览器下因为有此方法，所以可以直接获取到；  
	}  
	var nodes=document.getElementsByTagName("*");//获取页面里所有元素，因为他会匹配全页面元素，所以性能上有缺陷，但是可以约束他的搜索范围；  
	var arr=[];//用来保存符合的className；  
	for(var i=0;i<nodes.length;i++){  
		if(hasClass(nodes[i],className)) arr.push(nodes[i]);  
	}  
	return arr;  
}
var hasClass = function(node,className){  
	var cNames=node.className.split(/\s+/);//根据空格来分割node里的元素；  
	for(var i=0;i<cNames.length;i++){  
		if(cNames[i]==className) return true;  
	}  
	return false;  
}
var isIOS = function(){
	return /\((iPhone|iPad|iPod)/i.test(navigator.userAgent)?true:false;
}

var adNoneArr = getByClassName('ad-banner');
for(var i=0;i<adNoneArr.length;i++){
	if(adNoneArr[i].innerHTML == '' || (adNoneArr[i].innerHTML.toLowerCase().indexOf('.swf')!=-1&&isIOS())){
		adNoneArr[i].style.display = 'none';	
	}
}

setTimeout(function(){
//通栏01回收后给下方容器添加一个上边框线
if(document.getElementById('ad_25256')&&document.getElementById('ad_25256').style.display=='none'){
	document.getElementById('LejuText3').style.borderTop = '1px solid #E9E9E9';	
}
},1000);
  
	

})(document,window)