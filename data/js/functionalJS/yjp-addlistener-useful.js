<!--
YUE.addListener( document,  "keydown", YAHOO.Fp.KeyAction );
YUE.addListener( 'srchtxt', "keydown", function(e){if(e.keyCode == 38 || e.keyCode == 40 ) YAHOO.Fp.SearchAssist(e);} );
setInterval(function(e) {
  if(YAHOO.Fp._srchOldQ != $('srchtxt').value) {
    YAHOO.Fp._srchOldQ = $('srchtxt').value ;
    YAHOO.Fp.SearchAssist(e) ;
              }
} , 100) ;
YUE.addListener( 'srchAssistClose', "click", function(e){YUE.stopEvent(e);YAHOO.Fp.fToggleSearchAssist(e);} );
$('srchtxt').setAttribute("autocomplete", "off");
//-->