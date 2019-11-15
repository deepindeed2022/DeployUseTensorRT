#include <common/common.h>
#include <DataBlob.h>
#include <iomanip>

template <typename T>
DataBlob<T>::DataBlob():
    m_height(0), m_width(0), m_channel(0), m_step(-1),
    m_data(0), m_offset(0)
{
}
template <typename T>
DataBlob<T>::DataBlob(size_t height, size_t width, size_t channels, size_t step):
    m_height(height), m_width(width), m_channel(channels), m_step(step),
    m_data(new T[height * step], [](T *d) { delete[] d; }), m_offset(0)
{
    CHECK(step >= width * channels);
    CHECK(1 <= channels && channels <= 4);
    memset(m_data.get(), 0, sizeof(T) * height * step);
}

template <typename T>
DataBlob<T>::DataBlob(size_t height, size_t width, size_t channels):
    DataBlob(height, width, channels, width * channels)
{}

template <typename T>
DataBlob<T>::DataBlob(size_t height, size_t width, size_t channels, T *data):
    m_height(height), m_width(width), m_channel(channels),
    m_step(width * channels),
    m_data(data, [](T *) {}), m_offset(0)
{
}

template <typename T>
DataBlob<T>::DataBlob(size_t height, size_t width, size_t channels, size_t step, T *data):
    m_height(height), m_width(width), m_channel(channels),
    m_step(step),
    m_data(data, [](T *) {}), m_offset(0)
{
}

template <typename T>
DataBlob<T>::DataBlob(const DataBlob<T> &rhs):
    m_height(rhs.m_height), m_width(rhs.m_width), m_channel(rhs.m_channel),
    m_step(rhs.m_step),
    m_data(rhs.m_data), m_offset(0)
{
}

template <typename T>
DataBlob<T>::DataBlob(const DataBlob<T> &rhs,
        size_t row_offset, size_t row_count,
        size_t col_offset, size_t col_count):
    m_height(row_count), m_width(col_count), m_channel(rhs.m_channel),
    m_step(rhs.m_step), m_data(rhs.m_data),
    m_offset(rhs.m_offset + row_offset*m_step + col_offset*m_channel)
{
}

template <typename T>
DataBlob<T> &DataBlob<T>::operator=(const DataBlob<T> &rhs)
{
    this->m_height = rhs.m_height;
    this->m_width = rhs.m_width;
    this->m_channel = rhs.m_channel;
    this->m_step = rhs.m_step;
    this->m_data = rhs.m_data;
    this->m_offset = rhs.m_offset;
    return *this;
}

template <typename T>
T &DataBlob<T>::at(size_t r, size_t c, size_t ch)
{
    CHECK(r < m_height && c < m_width && ch < m_channel);
    return ptr(r)[c*m_channel + ch];
}

template <typename T>
const T &DataBlob<T>::at(size_t r, size_t c, size_t ch) const
{
    CHECK(r < m_height && c < m_width && ch < m_channel);
    return ptr(r)[c*m_channel + ch];
}

template <typename T>
DataBlob<T> DataBlob<T>::clone() const
{
    DataBlob<T> res(m_height, m_width, m_channel);
    for (size_t r = 0; r < m_height; ++r) {
        memcpy(res.ptr(r), this->ptr(r), sizeof(T) * m_width * m_channel);
    }
    return res;
}

template <typename T>
bool DataBlob<T>::equals(const DataBlob<T> &rhs) const
{
    if (this->m_height != rhs.m_height) return false;
    if (this->m_width != rhs.m_width) return false;
    if (this->m_channel != rhs.m_channel) return false;
    for (size_t r = 0; r < m_height; ++r) {
        if (0 != memcmp(this->ptr(r), rhs.ptr(r),
                    sizeof(T) * m_width * m_channel)) return false;
    }
    return true;
}

template <typename T>
bool DataBlob<T>::is_continuous() const
{ return m_step == m_width * m_channel; }

template <typename T>
void DataBlob<T>::read(const T *src)
{
    CHECK(is_continuous());
    memcpy(m_data.get(), src, sizeof(T) * this->total_nr_elem());
}

template <typename T>
void DataBlob<T>::write(T *dst) const
{
    CHECK(is_continuous());
    memcpy(dst, m_data.get(), sizeof(T) * this->total_nr_elem());
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const DataBlob<T> &m)
{
    for (size_t r = 0; r < m.height(); ++r) {
        for (size_t c = 0; c < m.width(); ++c) {
            os << '[';
            for (size_t ch = 0; ch < m.channels(); ++ch) {
                os << std::setw(4);
                os << static_cast<double>(m.at(r, c, ch));
                if (ch + 1 < m.channels()) os << ' ';
            }
            os << ']';
            if (c + 1 < m.width()) os << ',';
        }
        if (r + 1 < m.height()) os << std::endl;
    }
    return os;
}

template class DataBlob<uchar>;
template class DataBlob<float>;
template std::ostream &operator<<(std::ostream &os, const DataBlob<uchar> &m);
template std::ostream &operator<<(std::ostream &os, const DataBlob<float> &m);