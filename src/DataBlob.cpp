#include <common/common.h>
#include <DataBlob.h>
#include <iomanip>

template <typename T>
DataBlob<T>::DataBlob():
	m_num(0), m_channel(0), m_height(0), m_width(0), m_step(-1),
	m_data(0), m_offset(0)
{
}
template <typename T>
DataBlob<T>::DataBlob(size_t num, size_t height, size_t width, size_t channels):
	m_num(num), m_channel(channels), m_height(height), m_width(width),
	m_offset(0) 
{
	m_step = m_width*m_channel;
	m_data.reset(new T[m_num* m_height * m_step], [](T *d) { delete[] d; });
	memset(m_data.get(), 0, sizeof(T) * m_num* m_height * m_step);
}

template <typename T>
DataBlob<T>::DataBlob(DataBlobShape shape): 
	DataBlob(shape.nums(), shape.channels(), shape.heights(), shape.widths())
{}

template <typename T>
DataBlob<T>::DataBlob(size_t nums, size_t channels, size_t height, size_t width, T *data):
	m_num(nums), m_channel(channels), m_height(height), m_width(width),
	m_step(width * channels), m_data(data, [](T *) {}), m_offset(0)
{}

template <typename T>
DataBlob<T>::DataBlob(DataBlobShape shape, T *data):
	DataBlob(shape.nums(), shape.channels(), shape.heights(), shape.widths(), data)
{}

template <typename T>
DataBlob<T>::DataBlob(const DataBlob<T> &rhs):
	m_num(rhs.m_num),
	m_channel(rhs.m_channel),
	m_height(rhs.m_height), m_width(rhs.m_width), 
	m_step(rhs.m_step),
	m_data(rhs.m_data), m_offset(0)
{}

template <typename T>
DataBlob<T> &DataBlob<T>::operator=(const DataBlob<T> &rhs)
{
	this->m_num = rhs.m_num;
	this->m_channel = rhs.m_channel;
	this->m_height = rhs.m_height;
	this->m_width = rhs.m_width;
	this->m_step = rhs.m_step;
	this->m_data = rhs.m_data;
	this->m_offset = rhs.m_offset;
	return *this;
}

template <typename T>
T &DataBlob<T>::at(size_t n, size_t c, size_t h, size_t w)
{
	CHECK(h < m_height && w < m_width && c < m_channel && n < m_num);
	return ptr(n)[c*(m_width*m_height)+ h * m_width + w];
}

template <typename T>
const T &DataBlob<T>::at(size_t n, size_t c, size_t h, size_t w) const
{
	CHECK(h < m_height && w < m_width && c < m_channel && n < m_num);
	return ptr(n)[c*(m_width*m_height)+ h * m_width + w];
}

template <typename T>
DataBlob<T> DataBlob<T>::clone() const
{
	DataBlob<T> res(m_num, m_channel, m_height, m_width);
	for (size_t r = 0; r < m_num; ++r) {
		memcpy(res.ptr(r), this->ptr(r), sizeof(T) * this->inst_n_elem());
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
		if (0 != memcmp(this->ptr(r), rhs.ptr(r), sizeof(T) * inst_n_elem())) return false;
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
	memcpy(m_data.get(), src, sizeof(T) * this->total_n_elem());
}

template <typename T>
void DataBlob<T>::write(T *dst) const
{
	CHECK(is_continuous());
	memcpy(dst, m_data.get(), sizeof(T) * this->total_n_elem());
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const DataBlob<T> &m)
{
	// for (size_t r = 0; r < m.heights(); ++r) {
	// 	for (size_t c = 0; c < m.widths(); ++c) {
	// 		os << '[';
	// 		for (size_t ch = 0; ch < m.channels(); ++ch) {
	// 			os << std::setw(4);
	// 			os << static_cast<double>(m.at(r, c, ch));
	// 			if (ch + 1 < m.channels()) os << ' ';
	// 		}
	// 		os << ']';
	// 		if (c + 1 < m.widths()) os << ',';
	// 	}
	// 	if (r + 1 < m.heights()) os << std::endl;
	// }
	return os;
}

template class DataBlob<uchar>;
template class DataBlob<float>;
template std::ostream &operator<<(std::ostream &os, const DataBlob<uchar> &m);
template std::ostream &operator<<(std::ostream &os, const DataBlob<float> &m);