//    FEEDBURNER JAVASCRIPT LIBRARY
//    author: John Zeratsky <johnz@feedburner.com>
//    created: May 22, 2006
//    last modified: June 8, 2006

// 		!!! this must load after Sam's Prototype.js

	
	// Date.utcToLocal - converts gmt/utc date object to local time
	// call it on: a Date object in gmt/utc timezone
	// returns: a Date object in local time (local to the browser)

	Date.prototype.utcToLocal = function() {
	  	  
	  var translatedDate = new Date( Date.UTC( this.getFullYear(), this.getMonth(), this.getDate(), this.getHours(), this.getMinutes(), this.getSeconds(), this.getMilliseconds() ) );
	  return translatedDate;
	}


	// Element.nextElement and Element.prevElement - get the next or previous DOM elements (*not* the next or previous nodes, which includes text nodes; use native nextNode method for that)
	// accepts: a DOM node of type Element
	// returns: the next or previous DOM node of type Element

	Object.extend(Element, {
		
	  nextElement: function(el) {
	    var next = el;	
			do {
				next = next.nextSibling;
			} while( next.nodeType != 1 );
			return next;
	  },
	
		prevElement: function(el) {
			var prev = el;	
			do {
				prev = prev.previousSibling;
			} while( prev.nodeType != 1 );	
			return prev;
		},
		
		// Element.toggleReveal - toggle element and flip the little arrow image
		// accepts: Element object or id
		// returns: nuttin'
		
	  toggleReveal: function(el, link) {
			Element.toggle(el);
			if(link) {  		// if no link element was provided, it won't try to flip the arrow
				Element.removeClassName(link,"up");
				if(Element.visible(el)) Element.addClassName(link,"up");
			}
	  }
		
	});
