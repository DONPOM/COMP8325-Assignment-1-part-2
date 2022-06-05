	function toggle(el,tar) {
		
		var frame = document.getElementById(tar);
		frame.style.display=='none' ? frame.style.display='block' : frame.style.display='none';
		return false;	
	}
	
	function toggleSlide(el,tar) {
		
		var frame = document.getElementById(tar);
    frame.style.display=='none' ? Effect.BlindDown(tar) : Effect.BlindUp(tar);
		return false;	
	}
	
	function toggleThingAction(el) {

	  var elId = el.search.slice(1);
		Element.toggle(elId);
    
    return false;
	}
	
	function toggleDependentCheckbox( causer, effected ) {
  
	  if( $(causer).checked ) {
	    $(effected).disabled = false 
	  } else {
	    $(effected).disabled = true;
	    $(effected).checked = false;
	  }
	}

// new globalized functions

var FeedBurnerBase = {

  toggle: function(toggler) {
  	if( toggler.hash.length ) {
      Element.toggle(toggler.hash.slice(1));
      return false;      
    }
  },
  
  // gotta put this here for now
  popup: function(link) {
    window.open( link.href, 'popupwindow', 'scrollbars=yes,width=500,height=600');
    return false;
  }

};