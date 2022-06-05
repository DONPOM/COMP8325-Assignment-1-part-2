
    
        m3.util.History.init();
    
        function showCal(el) {
            cal = new m3.dotcom.widget.ArriveDepartTransit();
            cal.display({el:el});
        }
    
        function showCal2(el) {
            cal = new m3.dotcom.widget.RouteOptionsTransit();
            cal.display({el:el});
        }
    