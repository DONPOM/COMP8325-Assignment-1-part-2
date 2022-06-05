function findHeight(){try{var cspAd=parent.document.getElementById('csp-ad');if(cspAd){h=document.getElementById('member-ads').offsetHeight;if(h>250){h=250;}
h=h+3;cspAd.height=h;}}
catch(e){return;}}
window.onload=findHeight;