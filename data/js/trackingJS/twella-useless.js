
var exdoms = new Array("webcenter.polls.aol.com");
var oncedoms = new Array ("aim.com","ads.web.aol.com");
var url=''
try {url=document.referrer}
catch (e){}
url=url.substr(7);
url=url.toLowerCase();
var i=url.indexOf("/")
if (i==-1)i=url.length
var h=url.substr(0,i)
var doIt=1,dt=new Date(),oneVal=isInList(url,oncedoms)
if (oneVal){
 var c=readCk('atwTcSent')
 if (c){
    var arr=c.split('|')
    for (var i=0;i<arr.length;i++){
      if ((arr[i])==oneVal)doIt=0
    }
 }
 if (doIt==1){
  dt.setTime(dt.getTime()+(86400000)) 
  dt.setHours(0);
  dt.setMinutes(0);
  dt.setSeconds(0);
  if (c)c+='|'
  c+=oneVal
  document.cookie="atwTcSent="+c+"; expires="+dt.toGMTString()+"; path=/;" 
 }
}
if (doIt){
  if(!isInList(url,exdoms)) {
    var src;
    if (url.indexOf('huffingtonpost.com')>-1)
     src="http://an.tacoda.net/an/g10012/slf.js";
    else 
     src="http://an.tacoda.net/an/g10000/slf.js";
    document.write("<scr"+"ipt language='JavaScript' type='text/javascript' src='"+src+"'></scr"+"ipt>");
    var j=document.createElement('script');
    j.src='http://dtm.advertising.com/916f392e-1af6-43dd-bc12-239421d8b718.js';
    document.body.appendChild(j);
  }
}
var tacProp=new Object()
var d=location.hash
if (d){
  d=d.substr(1)
  var arr=d.split("&")
  for (var i=0; i<arr.length; i++){
    var kv=arr[i],kvs=kv.split("=")
    if (kvs[0]!=""){
      tacProp[kvs[0]]=unescape(kvs[1])
    }
  }
}
var qc='',ck='',qcCk='',qt='';
qcCk=readCk('tacDom')
if (qcCk.indexOf("quigob")>=0)ck='b';
else if (qcCk.indexOf("quigoa")>=0)ck='a';
else if (qcCk.indexOf("quigot")>=0)ck='t';
qc=readCk('atdemo');
qt=readCk('ATTAC');
if (!ck||(ck=='a'&&qt)||(ck=='t'&&qc))
{
 var i1=document.createElement('iframe')
 i1.id='qt1'
 if (!window.adsSaf)i1.style.display="none"
 i1.style.width='0px'
 i1.style.height='0px'
 i1.src='http://cdn.at.atwola.com/_media/uac/tcodeqt.html#qc'
 document.body.appendChild(i1)
}
function isInList(url,li){
 var l=li.length;
 for (var i=0; i<l; i++) {
   if(url.indexOf(li[i])>-1)return i+1;
 }
 return false;
}
function readCk(name){
 var c=document.cookie,b=c.indexOf(name+'=')
 if(b>=0){
  b=c.indexOf('=',b)+1
  if(b>0){
   var e=c.indexOf(';',b)
   if(e==-1)e=c.length
   return c.substring(b,e)
  }
 }
 return ''
}
