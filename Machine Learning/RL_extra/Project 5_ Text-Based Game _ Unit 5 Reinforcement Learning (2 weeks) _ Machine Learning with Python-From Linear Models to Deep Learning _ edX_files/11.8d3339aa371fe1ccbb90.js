(window.webpackJsonp=window.webpackJsonp||[]).push([[11],{975:function(e,n,a){"use strict";a.r(n);var t=a(0),i=a.n(t),r=a(1),s=a.n(r),o=a(31),m=a(21);function u(){return(u=Object.assign||function(e){for(var n=1;n<arguments.length;n++){var a=arguments[n];for(var t in a)Object.prototype.hasOwnProperty.call(a,t)&&(e[t]=a[t])}return e}).apply(this,arguments)}function c(e){var n,a=e.payload,t=a.description,r=a.endDate,s=a.userTimezone,c=s?{timeZone:s}:{},l=i.a.createElement(o.c,u({key:"timeRemaining",value:r},c));if(new Date(r)-new Date<864e5){var d=i.a.createElement(o.d,u({key:"courseEndTime",day:"numeric",month:"short",year:"numeric",timeZoneName:"short",value:r},c));n=i.a.createElement(o.b,{id:"learning.outline.alert.end.short",defaultMessage:"This course is ending {timeRemaining} at {courseEndTime}.",description:"Used when the time remaining is less than a day away.",values:{courseEndTime:d,timeRemaining:l}})}else{var p=i.a.createElement(o.a,u({key:"courseEndDate",day:"numeric",month:"short",year:"numeric",value:r},c));n=i.a.createElement(o.b,{id:"learning.outline.alert.end.long",defaultMessage:"This course is ending {timeRemaining} on {courseEndDate}.",description:"Used when the time remaining is more than a day away.",values:{courseEndDate:p,timeRemaining:l}})}return i.a.createElement(m.b,{type:m.a.INFO},i.a.createElement("strong",null,n),i.a.createElement("br",null),t)}c.propTypes={payload:s.a.shape({description:s.a.string,endDate:s.a.string,userTimezone:s.a.string}).isRequired},n.default=c}}]);
//# sourceMappingURL=11.8d3339aa371fe1ccbb90.js.map