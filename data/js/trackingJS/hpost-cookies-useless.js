
var dom=location.hash
if (dom!=''){
  dom=dom.substr(1)
  if (dom!=1)document.domain=dom
 
}

function adsPageOnL(){
  if (dom!=1){
try {  var adFr=window.frameElement } 
catch (e) { 
document.domain=document.domain;
var adFr=window.frameElement;
}
  if (adFr){
      if (adFr.textAd!=1){
        var collapse=parent.adsCkCol(adFr,document)
        if (!collapse&&adFr.divName){
          var parDiv=parent.document.getElementById(adFr.divName)
          if (parDiv&&(parDiv.dynSz==1)&&parent.adsRMIFOnL){
	      parent.adsRMIFOnL(window,document)
          }
        }
      }
      else
        parent.adsDoOnL(adFr,document)
  }
  }
}
