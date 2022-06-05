
    doWithAds(function(){
           window.jQuery && jQuery(function(){
              generic.document_is_ready()
           });
           generic.monitoring.stop_timing('page_load','',true);
           generic.monitoring.all_events_started();
       }, "No monitoring or document_is_ready object in generic");
