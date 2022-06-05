
var addthis_config = addthis_config||{};
    addthis_config.data_ga_tracker = "gaPageTracker";    addthis_config.data_track_addressbar = true;    addthis_config.data_ga_property = 'UA-1170033-1';
    addthis_config.data_ga_social = true;

    addthis.layers({
	    'theme' : 'transparent',
	    'share' : {
	      'position' : 'left',
	      'numPreferredServices' : 5
	    },
	    'follow':{
		    'label' : 'Follow:',
		    'services' : [
		        {'service' : 'facebook', 'id' : 'AddThis'},
		        {'service' : 'twitter', 'id' : 'AddThis'},
		        {'service' : 'google_follow', 'id' : '102383601500147943541'}
		    ],
		    'theme' : 'transparent'
		},
		'responsive' : true
    });
