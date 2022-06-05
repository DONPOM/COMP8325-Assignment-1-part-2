//jQuery.noConflicts();
		jQuery(document).ready(function(jQuery){
		
//Query String

function getParameters(){
    var
    settingsObject = {},
    hash,
    hashes =  location.search.substring(1).split(/&;/),
    i;
    for (i = 0; i < hashes.length; i++) {
        hash = hashes[i].split('=');
        settingsObject[hash[0]] = hash[1];
    }
    return settingsObject;
}
		
//Examples
	jQuery("a[rel='example1']").colorbox();
	jQuery("a[rel='example2']").colorbox({transition:"fade"});
	jQuery("a[rel='example3']").colorbox({transition:"none", width:"75%", height:"75%"});
	jQuery("a[rel='example4']").colorbox({slideshow:true});
	jQuery(".example6").colorbox({iframe:true, innerWidth:425, innerHeight:344});
	jQuery(".facility_map").colorbox();


//Adam Pop-ups
	jQuery(".adam").colorbox({width:"900", height:"80%", iframe:true});
//Art For Asthma Pop-ups
	jQuery(".artasthma_tall").colorbox({width:"720px", height:"98%", iframe:true});
	jQuery(".artasthma_wide").colorbox({width:"970px", height:"700px", iframe:true});
//Disclaimer Pop-up
	jQuery(".disclaimer").colorbox({width:"600px", height:"250px", iframe:true});
//Facility Homepage Pop-up Maps
	jQuery(".map_popup").colorbox({width:"900", height:"80%", iframe:true});
//BH Magazine Pop-up
	jQuery('a.bhmag_popup').colorbox(jQuery.extend({width:'820', height:'90%', iframe:true, speed:0}, getParameters()));	
//Timeline Pop-up
	jQuery(".timeline_popup").colorbox({width:"1037", height:"750", iframe:true});
//Beryl Pop-up
	jQuery('a.beryl').colorbox(jQuery.extend({width:'700', height:'400', iframe:true, speed:0}, getParameters()));
//FWFL Pop-up
	jQuery('a.fwfl_popup').colorbox(jQuery.extend({width:'700', height:'70%', iframe:true, speed:0}, getParameters()));
//IAGT Pop-up
	jQuery('a.iagt_popup').colorbox(jQuery.extend({width:'700', height:'70%', iframe:true, speed:0}, getParameters()));
//Email Pop-up
	jQuery('span.youtube').colorbox(jQuery.extend({iframe:true, innerWidth:425, innerHeight:344}, getParameters()));
//Univision Video Pop-up
	jQuery('.chefzgallery').colorbox(jQuery.extend({iframe:true, innerWidth:640, innerHeight:480}, getParameters()));
//Wax Ground Breaking Pop-up
	jQuery('a.wax-gb').colorbox(jQuery.extend({width:'500', height:'100%'}, getParameters()));




//More Examples	
			jQuery(".example8").colorbox({width:"50%", inline:true, href:"#inline_example1"});
			jQuery(".example9").colorbox({
				onOpen:function(){ alert('onOpen: colorbox is about to open'); },
				onLoad:function(){ alert('onLoad: colorbox has started to load the targeted content'); },
				onComplete:function(){ alert('onComplete: colorbox has displayed the loaded content'); },
				onCleanup:function(){ alert('onCleanup: colorbox has begun the close process'); },
				onClosed:function(){ alert('onClosed: colorbox has completely closed'); }
			});
			
			//Example of preserving a JavaScript event for inline calls.
			jQuery("#click").click(function(){ 
				jQuery('#click').css({"background-color":"#f00", "color":"#fff", "cursor":"inherit"}).text("Open this window again and this message will still be here.");
				return false;
			});
		});


