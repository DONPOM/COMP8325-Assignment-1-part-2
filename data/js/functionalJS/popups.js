var myrules = {
	'a.popup' : function(el) {
		element.onclick = function() {
			if( el.target == '_blank' ) {
			  return true;
			} else {
			  window.open( el.href, 'popupwindow', 'scrollbars=yes,width=500,height=600');
			  return false;
			}
		}
	},
	'a.popup600' : function(el) {
			element.onclick = function() {
				if( el.target == '_blank' ) {
				  return true;
				} else {
				  window.open( el.href, 'popupwindow', 'scrollbars=yes,width=600,height=600');
				  return false;
				}
			}
		},
		'a.popup660' : function(el) {
			element.onclick = function() {
				if( el.target == '_blank' ) {
				  return true;
				} else {
				  window.open( el.href, 'popupwindow', 'scrollbars=yes,width=660,height=600');
				  return false;
				}
			}
		}
};

Behaviour.register(myrules);