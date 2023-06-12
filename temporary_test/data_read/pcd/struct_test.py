import  struct
# native byteorder
buffer  =  struct.pack( "ihb" ,  1 ,  2 ,  3 )
print(buffer)
print  (repr ( buffer ))
print  (struct.unpack( "ihb" ,  buffer ))
# data from a sequence, network byteorder
data  =  [ 1 ,  2 ,  3 ]
buffer  =  struct.pack( "!ihb" ,  * data)
print(  repr ( buffer ))
print (struct.unpack( "!ihb" ,  buffer ))