
  var googletag = googletag || {};
  googletag.cmd = googletag.cmd || [];
  (function() {
    var gads = document.createElement("script");
    gads.async = true;
    gads.type = "text/javascript";
    var useSSL = "https:" == document.location.protocol;
    gads.src = (useSSL ? "https:" : "http:") + "//www.googletagservices.com/tag/js/gpt.js";
    var node =document.getElementsByTagName("script")[0];
    node.parentNode.insertBefore(gads, node);
   })();

   function bplGetChannels(channels) {

    var ad_channels = channels.split(",");
    var channel_conditions = {
      'top': 1626996852,
      'bottom': 3103730057,
      'sidebar': 4580463258,
      'homepage': 4695090852,
      'forums': 3218357652,
      'reviews': 6171824053,
      'articles': 7648557253,
      'popular': 2167100052,
      'rectangle': 6457698858,
      'dynamic': 7096827257
    };

    var channel_array = new Array();

    for ( var i = 0; i < ad_channels.length; i++){
      // if existing channel, replace with ID
      if (channel_conditions[ad_channels[i]])
        channel_array.push(channel_conditions[ad_channels[i]]);
      else
        // must be new channel, use the value passed through
        channel_array.push(ad_channels[i]);
    }
    return channel_array.join("+");
}

//http://bit.ly/1kYEXHQ
jQuery(document).ready(function(){
  var dfpslots=$("body").find(".adslot"),
  i = 0,
  slot = new Array();

  if (dfpslots.length) {
    googletag.cmd.push(function() {
      $(dfpslots).each(function(){

        //set ad unit vars
        if($(this).hasClass('leaderboard')) {
          var size = [728, 90];
          var altsize = [320, 50];
          var type = 'leaderboard';
        }
        if($(this).hasClass('rectangle')) {
          var size = [336, 280];
          var altsize = [300, 250];
          var type = 'rectangle';
        }
        if($(this).hasClass('skyscraper')) {
          var size = [160, 600];
          var altsize = [300, 250];
          var type = 'skyscraper';
        }

        //if larger than 940
        var mapping = googletag.sizeMapping().
          addSize([940,600], size).
          addSize([0,0], altsize).
          build();

        //define slots
        slot[i] = googletag.defineSlot('/119654936/'+$(this).attr('data-dfp'), size, $(this).attr('id')).defineSizeMapping(mapping).addService(googletag.pubads());
        //set channels and tergeting
        if ($(this).attr('data-section')) slot[i].set('adsense_channel_ids', bplGetChannels($(this).attr('data-section')+','+$(this).attr('data-position'))).
          setTargeting("position", $(this).attr('data-position')).
          setTargeting("type", type).
          setTargeting("section", $(this).attr('data-section').split(','));
          i++;
      });

      //googletag.pubads().enableSingleRequest(); // Breaks channel reporting
      googletag.enableServices();

      $(dfpslots).each(function(){
        googletag.display($(this).attr('id'));
      });
    });
  }
});
