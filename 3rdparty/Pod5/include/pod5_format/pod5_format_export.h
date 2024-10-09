
#ifndef POD5_FORMAT_EXPORT_H
#define POD5_FORMAT_EXPORT_H

#ifdef POD5_FORMAT_STATIC_DEFINE
#  define POD5_FORMAT_EXPORT
#  define POD5_FORMAT_NO_EXPORT
#else
#  ifndef POD5_FORMAT_EXPORT
#    ifdef pod5_format_EXPORTS
/* We are building this library */
#      define POD5_FORMAT_EXPORT 
#    else
/* We are using this library */
#      define POD5_FORMAT_EXPORT
#    endif
#  endif

#  ifndef POD5_FORMAT_NO_EXPORT
#    define POD5_FORMAT_NO_EXPORT
#  endif
#endif

#ifndef POD5_FORMAT_DEPRECATED
#  define POD5_FORMAT_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef POD5_FORMAT_DEPRECATED_EXPORT
#  define POD5_FORMAT_DEPRECATED_EXPORT POD5_FORMAT_EXPORT POD5_FORMAT_DEPRECATED
#endif

#ifndef POD5_FORMAT_DEPRECATED_NO_EXPORT
#  define POD5_FORMAT_DEPRECATED_NO_EXPORT POD5_FORMAT_NO_EXPORT POD5_FORMAT_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef POD5_FORMAT_NO_DEPRECATED
#    define POD5_FORMAT_NO_DEPRECATED
#  endif
#endif

#endif /* POD5_FORMAT_EXPORT_H */
