
include_directories(
  model
)

add_executable(main NNCNNLabeler.cpp)
message("libs:${LIBS}")

target_link_libraries(main ${LIBS})
