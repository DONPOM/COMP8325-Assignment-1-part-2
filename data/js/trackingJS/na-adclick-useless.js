
<!--
function DCFlash(id,pVM){
var swf = "http://s0.2mdn.net/4447375/AD021AU_MicroCFDsAUS200_728x90.swf";
var gif = "http://s0.2mdn.net/4447375/AD021AU_MicroCFDsAUS200_728x90.jpg";
var minV = 9;
var FWH = ' width="728" height="90" ';
var url = escape("http://adclick.g.doubleclick.net/pcs/click?xai=AKAOjss0Hbk3i08oUmRjxNOliBKUxbopTMA-dQRQ3pMuE6Ars7qy6au1ii5YkFC4WCK8-ochIZ0Ha6loIAHKFoGRXwzqc7_cdpaHVjZkplbXIyLDY5rVEohzt-tnGkzTwbw8j5PnTBg&sig=Cg0ArKJSzF6UCopV7azrEAE&adurl=http://www.forextrading.com.au/%3FCMP%3DSFS-70160000000DSBnAAO%26display%3DforexAU");
var fscUrl = url;
var fscUrlClickTagFound = false;
var wmode = "opaque";
var bg = "";
var dcallowscriptaccess = "never";

var openWindow = "false";
var winW = 0;
var winH = 0;
var winL = 0;
var winT = 0;

var moviePath=swf.substring(0,swf.lastIndexOf("/"));
var sm=new Array();


var defaultCtVal = escape("http://adclick.g.doubleclick.net/pcs/click?xai=AKAOjss0Hbk3i08oUmRjxNOliBKUxbopTMA-dQRQ3pMuE6Ars7qy6au1ii5YkFC4WCK8-ochIZ0Ha6loIAHKFoGRXwzqc7_cdpaHVjZkplbXIyLDY5rVEohzt-tnGkzTwbw8j5PnTBg&sig=Cg0ArKJSzF6UCopV7azrEAE&adurl=http://www.forextrading.com.au/%3FCMP%3DSFS-70160000000DSBnAAO%26display%3DforexAU");
var ctp=new Array();
var ctv=new Array();
ctp[0] = "clickTAG";
ctv[0] = "";


var fv='"moviePath='+moviePath+'/'+'&moviepath='+moviePath+'/';
for(i=1;i<sm.length;i++){if(sm[i]!=""){fv+="&submovie"+i+"="+escape(sm[i]);}}
for(var ctIndex = 0; ctIndex < ctp.length; ctIndex++) {
  var ctParam = ctp[ctIndex];
  var ctVal = ctv[ctIndex];
  if(ctVal != null && typeof(ctVal) == 'string') {
    if(ctVal == "") {
      ctVal = defaultCtVal;
    }
    else {
      ctVal = escape("http://adclick.g.doubleclick.net/pcs/click?xai=AKAOjss0Hbk3i08oUmRjxNOliBKUxbopTMA-dQRQ3pMuE6Ars7qy6au1ii5YkFC4WCK8-ochIZ0Ha6loIAHKFoGRXwzqc7_cdpaHVjZkplbXIyLDY5rVEohzt-tnGkzTwbw8j5PnTBg&sig=Cg0ArKJSzF6UCopV7azrEAE&adurl=" + ctVal);
    }
    if(ctParam.toLowerCase() == "clicktag") {
      fscUrl = ctVal;
      fscUrlClickTagFound = true;
    }
    else if(!fscUrlClickTagFound) {
      fscUrl = ctVal;
    }
    fv += "&" + ctParam + "=" + ctVal;
  }
}
fv+='"';
var bgo=(bg=="")?"":'<param name="bgcolor" value="#'+bg+'">';
var bge=(bg=="")?"":' bgcolor="#'+bg+'"';
function FSWin(){if((openWindow=="false")&&(id=="DCF0"))alert('openWindow is wrong.');
var dcw = 800;
var dch = 600;
// IE
if(!window.innerWidth)
{
  // strict mode
  if(!(document.documentElement.clientWidth == 0))
  {
    dcw = document.documentElement.clientWidth;
    dch = document.documentElement.clientHeight;
  }
  // quirks mode
  else if(document.body)
  {
    dcw = document.body.clientWidth;
    dch = document.body.clientHeight;
  }
}
// w3c
else
{
  dcw = window.innerWidth;
  dch = window.innerHeight;
}
if(openWindow=="center"){winL=Math.floor((dcw-winW)/2);winT=Math.floor((dch-winH)/2);}window.open(unescape(fscUrl),id,"width="+winW+",height="+winH+",top="+winT+",left="+winL+",status=no,toolbar=no,menubar=no,location=no");}this.FSWin = FSWin;
ua=navigator.userAgent;
if(minV<=pVM&&(openWindow=="false"||(ua.indexOf("Mac")<0&&ua.indexOf("Opera")<0))){
  var adcode='<object classid="clsid:d27cdb6e-ae6d-11cf-96b8-444553540000" id="'+id+'"'+FWH+'>'+
    '<param name="movie" value="'+swf+'"><param name="flashvars" value='+fv+'><param name="quality" value="high"><param name="wmode" value="'+wmode+'"><param name="base" value="'+swf.substring(0,swf.lastIndexOf("/"))+'"><PARAM NAME="AllowScriptAccess" VALUE="'+dcallowscriptaccess+'">'+bgo+
    '<embed src="'+swf+'" flashvars='+fv+bge+FWH+' type="application/x-shockwave-flash" quality="high" swliveconnect="true" wmode="'+wmode+'" name="'+id+'" base="'+swf.substring(0,swf.lastIndexOf("/"))+'" AllowScriptAccess="'+dcallowscriptaccess+'"></embed></object>';
  if(('i'!="j")&&(typeof dclkFlashWrite!="undefined")){dclkFlashWrite(adcode);}else{document.write(adcode);}
}else{
  document.write('<a target="_blank" href="'+unescape(url)+'"><img src="'+gif+'"'+FWH+'border="0" alt="Advertisement" galleryimg="no"></a>');
}}
function getFlashVersion() {
 var vfv="0,0,0";
 try {
  try {
   var axo=new ActiveXObject("ShockwaveFlash.ShockwaveFlash.6");
   try {
    axo.AllowScriptAccess="always";
   } catch(e) {
    return "6";
   }
  } catch(e) {}
  vfv=new ActiveXObject("ShockwaveFlash.ShockwaveFlash").GetVariable("$version");
 } catch(e) {
  try {
   if(navigator.mimeTypes["application/x-shockwave-flash"].enabledPlugin) {
    vfv=navigator.plugins["Shockwave Flash"].description;
   }
  } catch(e) {}
 }
 return vfv.replace(/\D+/g,",").match(/^,?(.+),?$/)[1].split(',').shift();
}
var DCid=(isNaN("283393331"))?"DCF2":"DCF283393331";
var pVM=getFlashVersion();
eval("function "+DCid+"_DoFSCommand(c,a){if(c=='openWindow')o"+DCid+".FSWin();}o"+DCid+"=new DCFlash('"+DCid+"',pVM);");
//-->
