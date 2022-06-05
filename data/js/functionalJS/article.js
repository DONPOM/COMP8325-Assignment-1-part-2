/**
 * A customized Backbone.Model that represents the data for an article used in the Ribbon
 *
 * <p><b>Require Path:</b> shared/data/models/article</p>
 *
 * @module Shared
 * @submodule Shared.Data
 * @namespace Data.SectionFeed
 * @class Model
 * @constructor
 * @extends Backbone.Model
**/
define('shared/data/models/article',[
    'underscore/nyt',
    'backbone/nyt'
], function (_, Backbone) {
    'use strict';

    var ArticleModel = Backbone.Model.extend({

        defaults: {
            title: '',
            kicker: '',
            pubdate: '',
            link: '',
            media: [{
                caption: '',
                'media-metadata': [{
                    url: ''
                }]
            }],
            promotional_media: [{
                image: {
                    image_crops: {}
                }
            }],
            processed: false
        },

        /**
         * Initializes the article model
         *
         * @private
         * @method initialize
        **/
        initialize: function () {
            this.adjustKicker();
        },

        /**
         * Adjusts the kicker according to specific rules
         *
         * @private
         * @method adjustKicker
        **/
        adjustKicker: function () {
            if(this.isOpinion() && this.get('authors')) {
                this.set('kicker', this.get('authors')[0]);
            }
        },

        /**
         * Checks whether the article is an opinion piece from a regular columnist
         *
         * @private
         * @method isOpinion
         * @return {Boolean} Whether the article is this kind of opinion piece
        **/
        isOpinion: function () {
            return (this.get('kicker') === 'Op-Ed Columnist' || this.get('kicker') === 'Opinionator');
        },

        /**
         * Get a specific image crop
         * @private
         * @method getCrop
         * @return image {Object} An object containing the image information
        **/
        getCrop: function (crop) {
            var media = this.get('promotional_media');
            var typeName = '';
            var type = 'image';

            //this feed seems to be unstable, wrap in a try-catch, so nothing breaks.
            try {
                //lede assets that are videos, slideshows, etc are in another promotional object
                if (!!media.type && media.type !== 'image') {

                    type = media.type;

                    switch (type) {
                        case 'video':
                            typeName = 'Play Video';
                            break;
                        case 'slideshow':
                            typeName = 'View Slideshow';
                            break;
                        case 'interactive':
                            typeName = 'View Interactive';
                            break;
                        default:
                            typeName = 'View Media';
                    }

                    media = media.promotional_media;
                }

                //add additional meta data to the payload
                media.image.image_crops[crop].type = type;
                media.image.image_crops[crop].typeName = typeName;

                return media.image.image_crops[crop];

            } catch (e) {
                return;
            }
        },

        /**
         *
         * This method will return the link/url attribute
         * @private
         * @method getLink
         * @return {String}
        **/
        getLink: function () {
            return this.get('link');
        },

        /**
         *
         *
         * @private
         * @method createDateObject
         * @param {String} dateString
         * @return {Object}
        **/
        createDateObject: function(dateString) {
            var yearMonthDate, yearMonthDateArray;
            var date = new Date(dateString);

            // fallback for cases where the browser does not support a certain format
            if (!date.getDate()) {

                // testing for ISO format
                if (dateString.search(/\d+T\d+/) >= 0) {
                    yearMonthDate = dateString.split('T')[0];
                    yearMonthDateArray = yearMonthDate.split('-');

                    date = new Date(yearMonthDateArray[0], (yearMonthDateArray[1] - 1), yearMonthDateArray[2]);
                }

            }

            return date;
        },

        /**
         * Getter for a date format that is usually displayed in story links, such as in marginalia
         * or story lists such as the Recommended Stories in Welcome Back
         *
         * @method getDateFormat
         * @param {String} datetime Provided datetime in a format different from article story display
         * @return {String} Date formatted for article story display
        **/
        getFormattedDate: function(datetime) {
            var date;
            var months = [
                'Jan.',
                'Feb.',
                'March',
                'April',
                'May',
                'June',
                'July',
                'Aug.',
                'Sept.',
                'Oct.',
                'Nov.',
                'Dec.'
            ];

            datetime = datetime || this.get('pubDate');
            date = this.createDateObject(datetime);

            return months[date.getMonth()] + ' ' + date.getDate() + ', ' + date.getFullYear();
        },

        /**
         * Getter for a datetime attribute format to be used with time HTML elements
         *
         * @method getDatetime
         * @param {String} datetime Provided datetime in a format different from datetime HTML attribute
         * @return {String} Datetime in HTML attribute format
        **/
        getDatetime: function(datetime) {
            var date;

            datetime = datetime || this.get('pubDate');
            date = this.createDateObject(datetime);

            return date.getFullYear() + '-' + ( date.getMonth() + 1 ) + '-' + date.getDate();
        }
    });

    return ArticleModel;
});

