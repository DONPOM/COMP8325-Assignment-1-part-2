function sendBooking() {
    var MinDate = new Date(document.getElementById(ASP_minDate).value);
    var MaxDate = new Date(document.getElementById(ASP_maxDate).value);

    if ((parseInt(document.getElementById(ASP_bookKids).value) + parseInt(document.getElementById(ASP_bookAdults).value)) > 4) {
        alert('There is a maximum of 4 people per room');
    } else {

        //check to see if the date is within the allowed date range:
        var arrivaldate = new Date(document.getElementById(ASP_bookArrival).value);

        if (arrivaldate < MinDate) {
            alert('The Arrival Date you entered is before the minimum date: ' + (MinDate.getMonth() + 1) + '/' + MinDate.getDate() + '/' + MinDate.getFullYear());

        } else if (arrivaldate > MaxDate) {
            alert('The Arrival Date you entered is after the maximum date: ' + (MaxDate.getMonth() + 1) + '/' + MaxDate.getDate() + '/' + MaxDate.getFullYear());
        } else {

            var departuredate = new Date(document.getElementById(ASP_bookDeparture).value);

            if (departuredate < MinDate) {
                alert('The Departure Date you entered is before the minimum date: ' + (MinDate.getMonth() + 1) + '/' + MinDate.getDate() + '/' + MinDate.getFullYear());

            } else if (departuredate > MaxDate) {
                alert('The Departure Date you entered is after the maximum date: ' + (MaxDate.getMonth() + 1) + '/' + MaxDate.getDate() + '/' + MaxDate.getFullYear());
            } else if (arrivaldate > departuredate) {
                alert('The Arrival Date you entered is after the Departure Date');
            } else {
                _blank = window.open('https://reservations.kbhmaui.com/reservations.htm?arrive=' + arrivaldate.getFullYear() + '-' + (arrivaldate.getMonth() + 1) + '-' + arrivaldate.getDate() + '&depart=' + departuredate.getFullYear() + '-' + (departuredate.getMonth() + 1) + '-' + departuredate.getDate() + '&adult=' + document.getElementById(ASP_bookAdults).value + '&child=' + document.getElementById(ASP_bookKids).value + '&rooms=1', '_blank', 'width=970,height=572,scrollbars=no,menubar=no, status=no, resizable=yes');
                _blank.focus();
            }
        }
    }
}

