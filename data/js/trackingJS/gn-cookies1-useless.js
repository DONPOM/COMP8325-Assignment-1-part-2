(function(isVeryModern){function insertUserName(){function getCookieValue(a){var d=[],e=document.cookie.split(";");a=RegExp("^\\s*"+a+"=\\s*(.*?)\\s*$");for(var b=0;b<e.length;b++){var f=e[b].match(a);f&&d.push(f[1]);}
if(d.length>0){return d[0];}
return null;}
function decodeBase64(str){return decodeURIComponent(encodeURIComponent(atob(str.replace(/-/g,'+').replace(/_/g,'/').replace(/,/g,'='))));}
function getUserDisplayNameFromCookie(){var cookieData=getCookieValue('GU_U');var userData=cookieData?JSON.parse(decodeBase64(cookieData.split('.')[0])):null;if(userData){return userData[2];}
return null;}
var userDisplayName=getUserDisplayNameFromCookie();if(userDisplayName){document.getElementsByClassName('js-profile-info')[0].innerHTML=userDisplayName;document.getElementsByClassName('js-profile-nav')[0].classList.add('is-signed-in');}}
insertUserName();})(guardian.isModernBrowser&&'atob'in window&&'classList'in document.documentElement);